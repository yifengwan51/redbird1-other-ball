"""MoTrack local web API.

Two tiers share one renderer:
- ``lightweight`` (free): the rule-based local editor.
- ``ai`` (VIP; ``llm`` is accepted as an alias): a real LLM decides the edit.
Binds to 127.0.0.1 only.  Phones reach it through a cloudflared tunnel;
anything that is not this machine's own browser must log in with the access
code (integrations/remote.py).  VIP is a local redemption-code gate.
"""
from __future__ import annotations
import json, os, re, subprocess, threading, uuid
from pathlib import Path
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from integrations import access, music, remote, uploads, vip
from integrations.security import validate_session_id

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs'; FF=ROOT/'tools/ffmpeg/bin/ffmpeg'; PROBE=ROOT/'tools/ffmpeg/bin/ffprobe'
THUMBS=OUT/'web_thumbs'
app=FastAPI(title='MoTrack'); jobs={}; lightweight_sessions={}; ai_sessions={}
app.mount('/app',StaticFiles(directory=ROOT/'web/frontend'),name='app')
BUILTIN={
 'djokovic':{'id':'djokovic','name':'Djokovic · US Open 2012','subtitle':'单打回合示例 · CC BY-SA 2.0','path':'inputs/tennis_djokovic_point.webm'},
 'davis-cup':{'id':'davis-cup','name':'Davis Cup · Uruguay vs Mexico','subtitle':'双打回合示例 · CC BY-SA 4.0','path':'inputs/tennis_davis_cup_doubles.webm'},
}
VIDEOS=dict(BUILTIN)  # built-ins + uploads that are ready to edit
def refresh_videos() -> None:
 fresh=dict(BUILTIN)
 for m in uploads.all_uploads():
  if m.get('status')=='ready' and uploads.video_path(m['id']).is_file():
   fresh[m['id']]={'id':m['id'],'name':m['title'],'subtitle':'我上传的视频','path':str(uploads.video_path(m['id']).relative_to(ROOT)),'upload':True}
 VIDEOS.clear(); VIDEOS.update(fresh)
 from integrations import lightweight
 lightweight.VIDEO_IDS.clear()
 lightweight.VIDEO_IDS.update({key: ROOT / value['path'] for key,value in VIDEOS.items()})
 for k in [k for k in _DURATIONS if k not in VIDEOS]: _DURATIONS.pop(k,None)
uploads.on_ready.append(refresh_videos)
MAX_JOBS=200
VARIANTS={'v1','v2','hypecut'}
API_VERSION=6  # the page warns when it talks to an older backend
vip.ensure_codes()
uploads.register_ready()
if os.getenv('MOTRACK_REMOTE')=='1': remote.tunnel.start()
if vip.dev_mode(): print('MoTrack: VIP 开发模式，所有用户默认是 VIP（MOTRACK_VIP_MODE=prod 关闭）',flush=True)

FRONTEND=ROOT/'web'/'frontend'
_DURATIONS={}
refresh_videos()

# ---------- access gate ----------
PUBLIC={'/app/login.html','/app/app.css','/app/theme.js','/app/admin-manifest.json','/app/manifest.json','/app/sw.js','/api/v1/login','/api/v1/health','/favicon.ico','/'}
def _headers(request:Request) -> dict: return {k.lower():v for k,v in request.headers.items()}
def _is_local(request:Request) -> bool: return not remote.is_remote(_headers(request))
def _signed_in(request:Request) -> bool: return _is_local(request) or remote.valid_session(request.cookies.get(remote.COOKIE))
def _owner(request:Request) -> bool:
 """The site owner: this computer, or a phone signed in with the access code / QR link."""
 return _signed_in(request)
def _visitor(request:Request) -> str|None: return getattr(request.state,'visitor',None)
def _ip(request:Request) -> str: return remote.client_ip(_headers(request),request.client.host if request.client else '')
ADMIN_ONLY=('/admin','/app/admin.html','/app/admin.js','/api/v1/admin/','/api/v1/remote')
# Paths that name a video: only its uploader (or the owner) may touch an upload.
_VIDEO_PATH=re.compile(r'^/api/v1/(?:videos|uploads)/(u[0-9a-f]{10})(?:/|$)')
def _can_see(request:Request, uid:str) -> bool:
 if uid in BUILTIN or _owner(request): return True
 meta=uploads.get(uid)
 return bool(meta) and meta.get('owner') is not None and meta.get('owner')==_visitor(request)
@app.middleware('http')
async def gate(request:Request, call_next):
 path=request.url.path
 h=_headers(request)
 token=request.cookies.get(access.COOKIE)
 request.state.visitor=access.visitor_id(token)
 fresh=None
 if request.state.visitor is None:
  fresh=access.new_token(); request.state.visitor=access.visitor_id(fresh)
 # Cross-site writes: a page on another site must not drive this API.
 if request.method not in ('GET','HEAD','OPTIONS') and h.get('origin'):
  origin=h['origin'].split('://',1)[-1].rstrip('/')
  if origin not in {h.get('host',''),h.get('x-forwarded-host','')}:
   return JSONResponse({'detail':'跨站请求被拒绝'},status_code=403)
 public=path in PUBLIC or path.startswith('/app/icons/') or path.startswith('/go/')
 admin_only=path.startswith(ADMIN_ONLY)
 if not public and not _signed_in(request) and (admin_only or not access.settings()['public']):
  if path.startswith('/api/'):
   if admin_only and access.settings()['public']:  # open site: not a login problem, just not the owner
    return JSONResponse({'detail':{'code':'ADMIN_ONLY','message':'只有管理员可以使用'}},status_code=403)
   return JSONResponse({'detail':{'code':'LOGIN_REQUIRED','message':'请先输入访问口令'}},status_code=401)
  return RedirectResponse('/app/login.html'+('?admin=1' if admin_only else ''),status_code=302)
 m=_VIDEO_PATH.match(path)
 if m and not _can_see(request,m.group(1)): return JSONResponse({'detail':'视频不存在'},status_code=404)
 response=await call_next(request)
 if fresh: response.set_cookie(access.COOKIE,fresh,max_age=access.COOKIE_DAYS*86400,httponly=True,
                              secure=remote.is_remote(h),samesite='lax',path='/')
 if path.startswith('/app/') and path.endswith(('.html','.js','.css','.json')): response.headers['Cache-Control']='no-cache'
 return response
def _cookie(response, request:Request) -> None:
 response.set_cookie(remote.COOKIE,remote.new_session(),max_age=remote.SESSION_DAYS*86400,httponly=True,
                     secure=remote.is_remote(_headers(request)),samesite='lax',path='/')

_locks_guard=threading.Lock(); _locks={}
def session_lock(sid:str) -> threading.Lock:
 """One request at a time per session: edits mutate shared plan state."""
 with _locks_guard: return _locks.setdefault(sid,threading.Lock())
def remember_job(jid:str,data:dict) -> dict:
 jobs[jid]=data
 finished=[k for k,v in jobs.items() if v.get('status') in ('completed','failed')]
 for k in finished[:max(0,len(jobs)-MAX_JOBS)]: jobs.pop(k,None)
 return data
def friendly(exc:Exception) -> str:
 """Show Chinese messages as-is; hide stack-trace style English ones."""
 text=str(exc)
 return text if re.search(r'[一-鿿]',text) else '剪辑时出了点问题，请稍后重试。'
def safe(p:Path):
 p=p.resolve()
 if ROOT not in p.parents and p!=ROOT: raise HTTPException(403,'拒绝项目外路径')
 return p
def read(p): return json.loads(safe(p).read_text())
def _session_id(payload:dict) -> str:
 raw=payload.get('session_id')
 if raw in (None,''): return uuid.uuid4().hex[:12]
 try: return validate_session_id(raw)
 except ValueError: raise HTTPException(400,'无效的会话')
def norm_mode(value) -> str:
 mode=str(value or 'lightweight')
 if mode in ('ai','llm'): raise HTTPException(400,'网球 AI 剪辑师正在适配中，请先使用基础剪辑')
 if mode!='lightweight': raise HTTPException(400,'不支持的模式')
 return mode
def require_vip(token:str|None) -> None:
 if not vip.is_vip(token): raise HTTPException(402,{'code':'VIP_REQUIRED','message':'AI 智能剪辑是 VIP 功能，开通后即可使用。'})
def use_ai(request:Request) -> None:
 """Count one AI use for a visitor; the owner is never limited."""
 if _owner(request): return
 try: access.use_ai(_visitor(request),_ip(request))
 except access.Denied as exc: raise HTTPException(403,{'code':exc.code,'message':exc.message})
def video_duration(video:str) -> float:
 if video in _DURATIONS: return _DURATIONS[video]
 try:
  raw=subprocess.check_output([str(PROBE),'-v','error','-show_entries','format=duration','-of','json',str(ROOT/VIDEOS[video]['path'])],text=True,timeout=30)
  _DURATIONS[video]=float(json.loads(raw)['format']['duration'])
  return _DURATIONS[video]
 except Exception:
  return 0.0
def event_score(event:dict) -> float:
 for key in ('confidence','final_ranking_score','raw_model_score','hypecut_score'):
  value=event.get(key)
  if isinstance(value,(int,float)): return float(value)
 return 0.0
def events(video:str, mode:str='lightweight', variant:str='v1') -> list[dict]:
 """Candidate moments shown on the page; the AI tier adds cached vision evidence."""
 if mode=='lightweight' and variant!='v1':
  from integrations.lightweight_v2 import candidates_for_variant
  return candidates_for_variant(video,variant)
 from integrations.lightweight import events as local_events
 items=local_events(video, analyze_if_missing=False)
 if mode=='ai':
  from integrations.llm_agent import vision_events, annotate
  items=annotate(video,items+vision_events(video))
 return items
def thumb(video:str, seconds:float, width:int, name:str) -> Path:
 """Cached JPEG frame; ffmpeg runs once per (video, time, size).

 The page asks for a dozen frames at once, so each name gets its own lock and
 every attempt its own temp file; concurrent requests never share a .part.
 """
 THUMBS.mkdir(parents=True,exist_ok=True)
 path=THUMBS/name
 with session_lock('thumb:'+name):
  if not path.is_file():
   part=path.with_name(f'.{uuid.uuid4().hex[:8]}_{name}')
   try:
    subprocess.run([str(FF),'-y','-v','error','-ss',f'{seconds:.2f}','-i',str(ROOT/VIDEOS[video]['path']),
                    '-frames:v','1','-vf',f'scale={width}:-2','-q:v','4','-f','image2',str(part)],check=True,timeout=30)
    os.replace(part,path)
   finally:
    part.unlink(missing_ok=True)
 return path

@app.get('/api/v1/health')
def health(): return {'status':'ok','api':API_VERSION,'modes':['lightweight'],'bind':'127.0.0.1'}
@app.get('/api/v1/capabilities')
def caps(x_vip_token:str|None=Header(default=None)):
 return {'lightweight':{'available':True},'ai':{'available':False,'reason':'网球版适配中','vip':False}}
@app.get('/api/v1/vip/status')
def vip_status(x_vip_token:str|None=Header(default=None)): return {'vip':vip.is_vip(x_vip_token),'dev':vip.dev_mode()}
@app.post('/api/v1/vip/activate')
def vip_activate(payload:dict):
 try: token=vip.activate(str(payload.get('code','')))
 except PermissionError: raise HTTPException(429,'尝试太频繁，请一分钟后再试')
 except ValueError: raise HTTPException(400,'兑换码无效，请检查后重试')
 return {'vip':True,'token':token}
@app.get('/api/v1/videos')
def videos(request:Request):
 out=[]
 owner=_owner(request); me=_visitor(request)
 for m in uploads.all_uploads():
  if not owner and (m.get('owner') is None or m.get('owner')!=me): continue
  if m.get('status')=='ready' and m['id'] in VIDEOS:
   out.append(VIDEOS[m['id']]|{'status':'ready','duration':round(video_duration(m['id']),3),'poster':f"/api/v1/videos/{m['id']}/poster",
                               'source':f"/api/v1/videos/{m['id']}/source",'vertical':(m.get('height') or 0)>(m.get('width') or 0)})
  else:
   out.append({'id':m['id'],'name':m.get('title','我的视频'),'subtitle':'我上传的视频','upload':True,'status':m.get('status'),
               'stage':m.get('stage'),'progress':m.get('progress',0),'error':m.get('error')})
 return out+[v|{'status':'ready','duration':round(video_duration(k),3),'poster':f'/api/v1/videos/{k}/poster','source':f'/api/v1/videos/{k}/source'}
             for k,v in BUILTIN.items()]
@app.get('/api/v1/videos/{video_id}/source')
def video_source(video_id:str):
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 path=safe(ROOT/VIDEOS[video_id]['path'])
 return FileResponse(path,media_type='video/webm' if path.suffix.lower()=='.webm' else 'video/mp4')

# ---------- uploads (chunked: the tunnel caps request bodies) ----------
@app.post('/api/v1/uploads')
def upload_start(payload:dict,request:Request):
 owner=_owner(request); me=_visitor(request)
 if not owner:
  limit=access.settings()['visitor_uploads']
  if sum(m.get('owner')==me for m in uploads.all_uploads())>=limit:
   raise HTTPException(400,f'每人最多保存 {limit} 个视频，请先删除一个' if limit else '暂时不开放上传')
 try: return uploads.start(str(payload.get('name','')),int(payload.get('size',0)),owner=None if owner else me)
 except (TypeError,ValueError) as exc: raise HTTPException(400,friendly(exc))
@app.put('/api/v1/uploads/{uid}')
async def upload_chunk(uid:str,request:Request,offset:int=0):
 data=await request.body()
 if len(data)>uploads.CHUNK+1024: raise HTTPException(413,'分片太大')
 try: return {'received':uploads.append(uid,offset,data)}
 except uploads.OffsetMismatch as exc: return JSONResponse({'detail':'offset','expected':exc.expected},status_code=409)
 except ValueError as exc: raise HTTPException(400,friendly(exc))
@app.post('/api/v1/uploads/{uid}/finish')
def upload_finish(uid:str):
 try: return uploads.finish(uid)
 except ValueError as exc: raise HTTPException(400,friendly(exc))
@app.get('/api/v1/uploads/{uid}')
def upload_status(uid:str):
 meta=uploads.get(uid)
 if not meta: raise HTTPException(404,'视频不存在')
 return meta
@app.delete('/api/v1/videos/{video_id}')
def video_delete(video_id:str):
 if video_id in BUILTIN: raise HTTPException(400,'示例视频不能删除')
 try: uploads.delete(video_id)
 except ValueError as exc: raise HTTPException(400,friendly(exc))
 for store in (lightweight_sessions,ai_sessions):
  for k in [k for k,v in store.items() if getattr(v,'video_id',None)==video_id]: store.pop(k,None)
 return {'deleted':video_id}

# ---------- visitors and the admin page ----------
@app.get('/admin')
def admin_page(): return FileResponse(FRONTEND/'admin.html',media_type='text/html',headers={'Cache-Control':'no-cache'})
@app.get('/api/v1/me')
def me(request:Request):
 cfg=access.settings()
 if _owner(request): return {'admin':True,'local':_is_local(request),'status':'owner','left':None,'free':cfg['free_ai_uses'],'public':cfg['public'],
                           'pending':access.listing()['counts']['pending']}
 return {'admin':False,'free':cfg['free_ai_uses'],'public':cfg['public'],**access.status(_visitor(request))}
@app.post('/api/v1/access/apply')
def access_apply(payload:dict,request:Request):
 if _owner(request): return me(request)
 try: return access.apply(_visitor(request),payload.get('name'),payload.get('note'),_ip(request))
 except PermissionError as exc: raise HTTPException(429,str(exc))
 except ValueError as exc: raise HTTPException(400,str(exc))
@app.get('/api/v1/admin/visitors')
def admin_visitors():
 data=access.listing()
 owned={}
 for m in uploads.all_uploads():
  if m.get('owner'): owned[m['owner']]=owned.get(m['owner'],0)+1
 for v in data['visitors']: v['uploads']=owned.get(v['id'],0)
 return data
@app.post('/api/v1/admin/visitors/{vid}')
def admin_decide(vid:str,payload:dict):
 try: return access.decide(vid,str(payload.get('action','')))
 except KeyError: raise HTTPException(404,'没有这个访客')
 except ValueError as exc: raise HTTPException(400,str(exc))
@app.post('/api/v1/admin/settings')
def admin_settings(payload:dict):
 try: return access.update_settings(payload)
 except ValueError as exc: raise HTTPException(400,str(exc))
@app.get('/api/v1/admin/site')
def admin_site(request:Request):
 """The public address to share.  Its QR code carries only the address, never the owner's login link."""
 st=remote.tunnel.status()
 url=st.get('url')
 out={**st,'url':url,'qr':remote.qr_svg(url) if url else None,'settings':access.settings(),'local':_is_local(request),
      'admin_url':f'{url}/admin' if url else None,'devices':len(remote._sessions())}
 if url and _is_local(request):  # the login QR is shown only on this computer's screen
  link=f'{url}/go/{remote.login_link_token()}?next=admin'
  out['admin_qr']=remote.qr_svg(link)
 return out
@app.post('/api/v1/admin/signout-all')
def admin_signout_all():
 """Sign out every phone and replace the QR login link (the access code stays)."""
 remote.reset_link(); return {'ok':True}

# ---------- phone access ----------
@app.get('/')
def root(): return RedirectResponse('/app/index.html',status_code=302)
@app.post('/api/v1/login')
def login(payload:dict,request:Request):
 h=_headers(request)
 try: ok=remote.check_code(str(payload.get('code','')),remote.client_ip(h,request.client.host if request.client else ''))
 except PermissionError: raise HTTPException(429,'尝试太频繁，请一分钟后再试')
 if not ok: raise HTTPException(400,'口令不对，请看电脑上的 MoTrack 页面')
 response=JSONResponse({'ok':True}); _cookie(response,request); return response
@app.get('/go/{token}')
def go(token:str,request:Request,next:str=''):
 if not remote.use_link(token): return RedirectResponse('/app/login.html?reset=1'+('&admin=1' if next=='admin' else ''),status_code=302)
 response=RedirectResponse('/admin' if next=='admin' else '/app/index.html',status_code=302); _cookie(response,request); return response
@app.get('/api/v1/remote')
def remote_info(request:Request):
 if not _is_local(request): raise HTTPException(403,'只能在电脑上查看')
 info={'available':remote.CLOUDFLARED.is_file(),'code':remote.pretty_code(),**remote.tunnel.status()}
 if info['url']:
  link=f"{info['url']}/go/{remote.login_link_token()}"
  info.update(link=link,qr=remote.qr_svg(link))
 return info
@app.post('/api/v1/remote/reset')
def remote_reset(request:Request):
 if not _is_local(request): raise HTTPException(403,'只能在电脑上重置')
 remote.reset_link(); return remote_info(request)
@app.post('/api/v1/remote/start')
def remote_start(request:Request):
 if not _is_local(request): raise HTTPException(403,'只能在电脑上开启')
 remote.tunnel.start(); return remote_info(request)
@app.post('/api/v1/remote/stop')
def remote_stop(request:Request):
 if not _is_local(request): raise HTTPException(403,'只能在电脑上关闭')
 remote.tunnel.stop(); return remote_info(request)
@app.get('/api/v1/videos/{video_id}/poster')
def poster(video_id:str):
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 at=min(12.0,max(0.0,video_duration(video_id)*0.3))
 try: return FileResponse(thumb(video_id,at,640,f'{video_id}_poster.jpg'),headers={'Cache-Control':'max-age=86400'})
 except Exception: raise HTTPException(404,'封面暂不可用')
@app.get('/api/v1/videos/{video_id}/frame')
def frame(video_id:str,t:float):
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 length=video_duration(video_id)
 if not 0<=t<=max(0.0,length): raise HTTPException(400,'时间超出原片')
 at=round(t*2)/2  # half-second grid keeps the cache small
 try: return FileResponse(thumb(video_id,at,320,f'{video_id}_{int(at*10):06d}.jpg'),headers={'Cache-Control':'max-age=86400'})
 except Exception: raise HTTPException(404,'画面暂不可用')
@app.get('/api/v1/videos/{video_id}/events')
def get_events(video_id:str,mode:str='lightweight',variant:str='v1'):
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 mode=norm_mode(mode)
 if variant not in VARIANTS: raise HTTPException(400,'不支持的候选版本')
 items=events(video_id,mode,variant)
 return {'events':items,'mode':mode,'shooting':None}
@app.get('/api/v1/videos/{video_id}/review')
def get_review(video_id:str):
 """复盘: timeline, per-period shooting and streaks, from the cached analysis."""
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 from integrations import review
 items=events(video_id,'lightweight','v1')
 if not items: raise HTTPException(409,'这段视频还没分析过，先点一次「开始剪辑」')
 report=review.build(video_id,items,video_duration(video_id))
 report['recap']=_cached_recap(video_id,review.recap_input(report))
 return report
@app.post('/api/v1/videos/{video_id}/review/correct')
def correct_shot(video_id:str, request:Request, payload:dict):
 """Admin-only label fix for evaluation (null = back to the detector's call). Not shown to users."""
 if not _owner(request): raise HTTPException(403,'只有管理员可以使用')
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 from integrations import corrections
 from integrations.lightweight import _raw_events
 eid=str(payload.get('event_id','')); want=payload.get('outcome')
 orig=next((e for e in _raw_events(video_id) if e.get('event_id')==eid and e.get('evidence_source')=='LOCAL_DETECTOR'),None)
 if orig is None: raise HTTPException(404,'找不到这次出手')
 try: corrections.set_outcome(video_id,eid,want if want in ('made','missed') else None,orig.get('outcome'))
 except ValueError as exc: raise HTTPException(400,str(exc))
 return get_review(video_id)
@app.get('/api/v1/videos/{video_id}/players')
def get_players(video_id:str):
 """Numbered players (by shots taken) with their shooting; empty until analysed."""
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 from integrations.lightweight import roster
 r=roster(video_id)
 return {'video_id':video_id,'teams':r.get('teams',False),
         'players':[{k:p[k] for k in ('id','number','team','attempts','made','missed','fg_pct')} for p in r.get('players',[])]}
@app.get('/api/v1/videos/{video_id}/players/{pid}/photo')
def player_photo(video_id:str, pid:str):
 if video_id not in VIDEOS or not re.fullmatch(r'p\d{1,3}',pid): raise HTTPException(404,'未知球员')
 from integrations.lightweight import roster
 p=next((x for x in roster(video_id).get('players',[]) if x['id']==pid),None)
 if p is None: raise HTTPException(404,'未知球员')
 t=float(p['thumb']['t']); x1,y1,x2,y2=p['thumb']['box']
 name=f"pl_{video_id}_{pid}_{int(t*100)}_{int(x1)}_{int(y1)}.jpg"
 path=THUMBS/name
 if not path.exists():
  import cv2
  THUMBS.mkdir(parents=True,exist_ok=True)
  cap=cv2.VideoCapture(str(ROOT/VIDEOS[video_id]['path'])); cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,f=cap.read(); cap.release()
  if not ok: raise HTTPException(404,'取不到画面')
  H,W=f.shape[:2]; w,h=x2-x1,y2-y1; pad=0.25
  crop=f[int(max(0,y1-pad*h)):int(min(H,y2+pad*h)),int(max(0,x1-pad*w)):int(min(W,x2+pad*w))]
  if crop.size==0: raise HTTPException(404,'取不到画面')
  s_=200/max(crop.shape[:2]); crop=cv2.resize(crop,(max(1,int(crop.shape[1]*s_)),max(1,int(crop.shape[0]*s_))),interpolation=cv2.INTER_CUBIC)
  part=path.with_suffix('.part.jpg'); cv2.imwrite(str(part),crop,[cv2.IMWRITE_JPEG_QUALITY,88]); os.replace(part,path)
 return FileResponse(path,media_type='image/jpeg',headers={'Cache-Control':'max-age=3600'})
REVIEW_DIR=ROOT/'outputs'/'review'
_recap_lock=threading.Lock()
def _recap_key(text:str) -> str:
 import hashlib
 return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
def _cached_recap(video_id:str,text:str) -> str|None:
 try: data=json.loads((REVIEW_DIR/f'{video_id}_recap.json').read_text(encoding='utf-8'))
 except (OSError,ValueError): return None
 return data.get('recap') if data.get('key')==_recap_key(text) else None
@app.post('/api/v1/videos/{video_id}/review/recap')
def make_recap(video_id:str, request:Request, x_vip_token:str|None=Header(default=None)):
 """VIP: the AI writes a short coach-style recap from the statistics only."""
 require_vip(x_vip_token)
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 from integrations import review
 items=events(video_id,'lightweight','v1')
 if not items: raise HTTPException(409,'这段视频还没分析过，先点一次「开始剪辑」')
 report=review.build(video_id,items,video_duration(video_id))
 text=review.recap_input(report)
 with _recap_lock:
  if (hit:=_cached_recap(video_id,text)): return {'recap':hit,'cached':True}
  use_ai(request)
  try:
   from openai import OpenAI
   from integrations.api_config import resolve_base_url
   from integrations.llm_agent import api_key, DEFAULT_MODEL
   client=OpenAI(api_key=api_key(),base_url=resolve_base_url(),timeout=60,max_retries=1)
   reply=client.chat.completions.create(model=DEFAULT_MODEL,max_tokens=500,temperature=0.3,extra_body={'reasoning_effort':'low'},
    messages=[{'role':'user','content':review.recap_prompt(report)}])
   recap=(reply.choices[0].message.content or '').strip()
  except Exception as exc:
   raise HTTPException(502,'AI 复盘暂时写不出来，稍后再试') from exc
  if not recap: raise HTTPException(502,'AI 复盘暂时写不出来，稍后再试')
  REVIEW_DIR.mkdir(parents=True,exist_ok=True)
  part=REVIEW_DIR/f'.{video_id}_recap.part'
  part.write_text(json.dumps({'key':_recap_key(text),'recap':recap},ensure_ascii=False),encoding='utf-8')
  os.replace(part,REVIEW_DIR/f'{video_id}_recap.json')
 return {'recap':recap,'cached':False}
@app.get('/api/v1/videos/{video_id}/review/export')
def export_review(video_id:str):
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 from integrations import review
 items=events(video_id,'lightweight','v1')
 if not items: raise HTTPException(409,'这段视频还没分析过，先点一次「开始剪辑」')
 report=review.build(video_id,items,video_duration(video_id))
 page=review.to_html(report,VIDEOS[video_id]['name'],_cached_recap(video_id,review.recap_input(report)))
 from urllib.parse import quote
 return Response(page,media_type='text/html; charset=utf-8',
  headers={'Content-Disposition':"attachment; filename*=UTF-8''"+quote(VIDEOS[video_id]['name']+'_复盘.html')})
@app.get('/api/v1/videos/{video_id}/candidates')
def candidates(video_id:str,variant:str='v2'):
 if video_id not in VIDEOS or variant not in VARIANTS: raise HTTPException(404,'未知视频或版本')
 return {'video_id':video_id,'candidate_version':variant,'events':events(video_id,'lightweight',variant)}
@app.get('/api/v1/music')
def music_list():
 presets=[{'id':k,'title':v['title'],'desc':v['desc'],'bpm':v['bpm']} for k,v in music.PRESETS.items()]
 uploads=[]
 if music.UPLOADS.is_dir():
  for meta_file in sorted(music.UPLOADS.glob('*.json'),key=lambda f:f.stat().st_mtime,reverse=True)[:20]:
   meta=music.upload_meta(meta_file.stem)
   if meta: uploads.append({'id':meta['id'],'title':meta['title'],'duration':meta['duration']})
 return {'presets':presets,'uploads':uploads}
@app.get('/api/v1/music/{music_id}/preview')
def music_preview(music_id:str):
 if not music.valid_choice(music_id) or music_id=='none': raise HTTPException(404,'没有这首音乐')
 track,_=music.resolve(music_id)
 if not track: raise HTTPException(404,'没有这首音乐')
 return FileResponse(track,headers={'Cache-Control':'max-age=86400'})
@app.post('/api/v1/music/upload')
async def music_upload(file:UploadFile=File(...)):
 name=Path(file.filename or 'music.mp3').name
 if Path(name).suffix.lower() not in music.AUDIO_SUFFIXES: raise HTTPException(415,'仅支持 MP3 / M4A / AAC / WAV / OGG / FLAC')
 music.UPLOADS.mkdir(parents=True,exist_ok=True)
 tmp=music.UPLOADS/f'.{uuid.uuid4().hex[:8]}{Path(name).suffix.lower()}'
 size=0
 try:
  with tmp.open('wb') as out:
   while True:
    chunk=await file.read(1024*1024)
    if not chunk: break
    size+=len(chunk)
    if size>music.MAX_UPLOAD_BYTES: raise HTTPException(413,'音乐文件超过 30MB')
    out.write(chunk)
  try: meta=music.save_upload(tmp,name)
  except ValueError as exc: raise HTTPException(400,str(exc))
 finally:
  tmp.unlink(missing_ok=True)
 return {'id':meta['id'],'title':meta['title'],'duration':meta['duration']}
@app.get('/api/v1/media/{kind}/{name}')
def media(kind:str,name:str):
 base={'source':ROOT/'inputs','job':OUT/'web_jobs'}.get(kind)
 if not base or '/' in name or '\\' in name: raise HTTPException(400,'无效资源名')
 p=safe(base/name)
 if not p.is_file(): raise HTTPException(404,'文件不存在')
 return FileResponse(p)
@app.get('/api/v1/artifact')
def artifact(path:str,dl:int=0):
 p=safe(ROOT/path)
 allowed=(OUT/'lightweight_agent', OUT/'lightweight', OUT/'lightweight_quality_v2', OUT/'web_jobs')
 if not any(base.resolve() in (p, *p.parents) for base in allowed): raise HTTPException(403,'不允许的产物路径')
 if not p.is_file() or p.suffix.lower() not in {'.mp4','.json'}: raise HTTPException(404,'产物不存在或类型不允许')
 if dl and p.suffix.lower()=='.mp4':
  return FileResponse(p,media_type='video/mp4',headers={'Content-Disposition':"attachment; filename=\"MoTrack.mp4\"; filename*=UTF-8''"+quote('MoTrack集锦.mp4')})
 return FileResponse(p)
def ai_editor(video_id:str, sid:str):
 from integrations.llm_agent import LLMEditor
 editor=ai_sessions.get(sid)
 if editor is None:
  editor=ai_sessions[sid]=LLMEditor(video_id,sid,video_name=VIDEOS[video_id]['name'])
 elif editor.video_id!=video_id: raise HTTPException(409,'这段对话属于另一段视频，请点“新对话”')
 return editor

@app.post('/api/v1/highlights')
def highlights(payload:dict, request:Request, x_vip_token:str|None=Header(default=None)):
 vid=payload.get('video_id'); types=payload.get('event_types',['tennis_action_candidate'])
 try: duration=float(payload.get('target_duration',15))
 except (TypeError,ValueError): raise HTTPException(400,'目标时长无效')
 mode=norm_mode(payload.get('mode'))
 variant=payload.get('candidate_version','v1')
 style=payload.get('edit_style') or {}
 if vid not in VIDEOS:
  meta=uploads.get(str(vid))
  raise HTTPException(409 if meta else 404,'视频还在处理中，稍等一下' if meta and meta.get('status')!='failed' else '未知视频')
 if not 5<=duration<=60 or not isinstance(types,list): raise HTTPException(400,'时长需在 5–60 秒之间')
 if variant not in VARIANTS: raise HTTPException(400,'不支持的候选版本')
 serves='serve' in types
 tennis='tennis_hit' in types or serves
 if mode=='ai':
  require_vip(x_vip_token); use_ai(request)
  jid='ai-'+uuid.uuid4().hex[:12]
  job=remember_job(jid,{'id':jid,'mode':'ai','status':'queued','stage':'QUEUED','session_id':jid,'video_id':vid})
  def ai_work():
   job.update(status='running',stage='AI_EDITING')
   try:
    with session_lock(jid):
     editor=ai_editor(vid,jid)
     editor.session.progress=lambda done,total: job.update(stage='ANALYZING',progress=round(100*done/max(1,total)))
     editor.session.stage_hook=lambda name: job.update(stage=name)
     result=editor.generate(duration,'shots' if tennis else 'all',style)
     editor.session.stage_hook=None  # later chat turns have no job to report to
    job.update(status='completed',stage='OUTPUT_VERIFIED',result=result,progress=100)
   except Exception as exc:
    job.update(status='failed',stage='FAILED',error={'message':friendly(exc)})
  threading.Thread(target=ai_work,daemon=True).start()
  return job
 from integrations.lightweight import LightweightSession
 jid='lw-'+uuid.uuid4().hex[:12]
 session=LightweightSession(video_id=vid,session_id=jid,candidate_version=variant); lightweight_sessions[jid]=session
 prompt=('生成发球与击球集锦 ' if (tennis and variant!='hypecut') else '生成网球高光 ')+f'{duration:g}秒'
 job=remember_job(jid,{'id':jid,'mode':'lightweight','candidate_version':variant,'status':'queued','stage':'QUEUED','session_id':jid,'video_id':vid,'prompt':prompt})
 session.progress=lambda done,total: job.update(stage='ANALYZING',progress=round(100*done/max(1,total)))
 session.stage_hook=lambda name: job.update(stage=name)
 def local_work():
  job.update(status='running',stage='LOCAL_AGENT_RUNNING')
  try:
   with session_lock(jid): result=session.ask(prompt,edit_style=style)
   session.stage_hook=None
  except Exception as exc:
   job.update(status='failed',stage='FAILED',error={'message':friendly(exc)}); return
  if result.get('failure'):
   job.update(status='failed',stage='FAILED',error={'message':result.get('reply','生成失败')},result=result)
  else:
   out=result.get('output') or {}
   job.update(status='completed',stage='OUTPUT_VERIFIED',result=result,progress=100,
              requested_duration=out.get('requested_duration',duration),effective_duration=out.get('effective_duration'),
              duration_clamped=out.get('duration_clamped',False),duration_note=out.get('duration_note',''))
 threading.Thread(target=local_work,daemon=True).start()
 return job
@app.get('/api/v1/jobs/{job_id}')
def job(job_id:str):
 if job_id not in jobs: raise HTTPException(404,'任务不存在')
 return jobs[job_id]
@app.get('/api/v1/jobs/{job_id}/trace')
def trace(job_id:str):
 if job_id not in jobs: raise HTTPException(404,'任务不存在')
 path=OUT/'lightweight_agent'/job_id/'tool_trace.json'
 if not path.is_file(): raise HTTPException(404,'轨迹尚未生成')
 return read(path)
def _segment(payload:dict) -> int|None:
 try: n=int(payload.get('selected_segment') or 0)
 except (TypeError,ValueError): return None
 return n if 1<=n<=50 else None
@app.post('/api/v1/chat')
def chat(payload:dict, request:Request, x_vip_token:str|None=Header(default=None)):
 text=str(payload.get('message','')).strip()
 if not text: raise HTTPException(400,'消息不能为空')
 if len(text)>500: raise HTTPException(400,'消息太长了，请精简到 500 字以内')
 mode=norm_mode(payload.get('mode'))
 video_id=str(payload.get('video_id',''))
 variant=str(payload.get('candidate_version','v1'))
 if video_id not in VIDEOS: raise HTTPException(404,'未知视频')
 sid=_session_id(payload)
 if mode=='ai':
  require_vip(x_vip_token); use_ai(request)
  with session_lock(sid):
   try: return ai_editor(video_id,sid).turn(text,edit_style=payload.get('edit_style'),selected_segment=_segment(payload))
   except HTTPException: raise
   except Exception as exc: raise HTTPException(422,friendly(exc))
 from integrations.lightweight import LightweightSession
 if variant not in VARIANTS: raise HTTPException(400,'不支持的候选版本')
 with session_lock(sid):
  try:
   if sid not in lightweight_sessions: lightweight_sessions[sid]=LightweightSession(video_id=video_id,session_id=sid,candidate_version=variant)
   elif lightweight_sessions[sid].video_id!=video_id: raise HTTPException(409,'这段对话属于另一段视频，请点“新对话”')
   result=lightweight_sessions[sid].ask(text,edit_style=payload.get('edit_style'),selected_segment=_segment(payload))
   if result.get('failure'): result['error']=result['failure']
   return result
  except HTTPException: raise
  except Exception as exc: raise HTTPException(422,friendly(exc))
