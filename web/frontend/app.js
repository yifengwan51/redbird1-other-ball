(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => Array.from(document.querySelectorAll(s));
  const TOKEN_KEY = 'motrack_vip_token';
  const API_VERSION = 6;
  const DURATIONS = [10, 15, 20, 30, 45, 60];
  const STYLE_KEYS = ['pace', 'transition', 'look', 'zoom', 'music', 'slow_motion'];
  const PRESETS = [
    { id: 'hype', cls: 'p-hype', emo: '🔥', name: '燃炸快剪', desc: '热血配乐·闪白·卡点',
      style: { pace: 'tight', transition: 'flash', look: 'vivid', zoom: true, music: 'hype', slow_motion: true } },
    { id: 'cinema', cls: 'p-cinema', emo: '🎬', name: '电影大片', desc: '宽银幕·推镜·慢动作',
      style: { pace: 'relaxed', transition: 'fade', look: 'cinematic', zoom: true, music: 'chill', slow_motion: true } },
    { id: 'street', cls: 'p-street', emo: '🎤', name: '街头嘻哈', desc: '嘻哈律动·硬切',
      style: { pace: 'standard', transition: 'cut', look: 'vivid', zoom: false, music: 'groove', slow_motion: false } },
    { id: 'retro', cls: 'p-retro', emo: '📼', name: '复古胶片', desc: '胶片调色·律动',
      style: { pace: 'standard', transition: 'fade', look: 'retro', zoom: false, music: 'groove', slow_motion: false } },
    { id: 'bw', cls: 'p-bw', emo: '🖤', name: '黑白经典', desc: '黑白·舒缓·推镜',
      style: { pace: 'relaxed', transition: 'fade', look: 'bw', zoom: true, music: 'chill', slow_motion: true } },
    { id: 'raw', cls: 'p-raw', emo: '🎾', name: '原汁原味', desc: '保留现场声·不加特效',
      style: { pace: 'standard', transition: 'fade', look: 'vivid', zoom: false, music: 'none', slow_motion: false } }
  ];
  // Progress steps shown on the player while a job runs; stage -> step index.
  const FLOWS = {
    lightweight: { steps: ['看比赛', '挑镜头', '剪辑渲染'],
                   at: { QUEUED: 0, ANALYZING: 0, LOCAL_AGENT_RUNNING: 1, RENDERING: 2, OUTPUT_VERIFIED: 3 } },
    ai: { steps: ['看比赛', 'AI 看画面', '剪辑渲染', '写解说'],
          at: { QUEUED: 0, ANALYZING: 0, AI_EDITING: 1, AI_LOOKING: 1, RENDERING: 2, AI_WRITING: 3, OUTPUT_VERIFIED: 4 } }
  };
  const STAGE_TEXT = {
    QUEUED: '排队中…', ANALYZING: '第一次处理这段视频，正在看比赛…', LOCAL_AGENT_RUNNING: '正在挑选精彩镜头…',
    AI_EDITING: 'AI 剪辑师开始工作…', AI_LOOKING: 'AI 正在看画面，挑真正精彩的镜头…', RENDERING: '正在剪辑、加配乐和特效…',
    AI_WRITING: 'AI 正在写解说…', OUTPUT_VERIFIED: '剪好了'
  };
  const SUGGEST = {
    lightweight: ['缩短到 10 秒', '换一首', '改成竖屏', '只保留 3 段', '做成电影感', '加特效'],
    ai: ['把没看点的镜头去掉', '最精彩的是哪一拍？', '只要后半段', '改成竖屏，节奏快一点', '换成复古，配乐换嘻哈', '标题改成“球场之夜”']
  };

  const state = { videos: [], video: null, mode: 'lightweight', vip: false, dev: false, duration: 20,
                  preset: 'hype', sessionId: '', busy: false, output: null, events: [],
                  music: 'hype', musicList: [], titleOff: false, pane: 'edit' };

  const store = {
    get() { try { return localStorage.getItem(TOKEN_KEY) || ''; } catch (e) { return ''; } },
    set(v) { try { v ? localStorage.setItem(TOKEN_KEY, v) : localStorage.removeItem(TOKEN_KEY); } catch (e) { /* private mode */ } }
  };
  const esc = (t) => String(t == null ? '' : t).replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const clock = (s) => { s = Math.max(0, Number(s) || 0); return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'); };
  const frameUrl = (t) => '/api/v1/videos/' + state.video.id + '/frame?t=' + Math.max(0, Math.min(t, (state.video.duration || 1) - 0.1)).toFixed(1);
  const artifactUrl = (path, bust) => '/api/v1/artifact?path=' + encodeURIComponent(path) + (bust ? '&v=' + Date.now() : '');
  const narrow = () => window.matchMedia('(max-width: 899px)').matches;
  // <img> inside a shimmering placeholder; a failed load leaves a clean gradient.
  const img = (url) => '<img loading="lazy" decoding="async" alt="" src="' + esc(url) + '" onload="this.parentNode.classList.add(\'ok\')" onerror="this.parentNode.classList.add(\'bad\')">';

  class ApiError extends Error { constructor(message, status, code) { super(message); this.status = status; this.code = code; } }
  async function api(path, body) {
    const headers = {};
    const token = store.get();
    if (token) headers['X-VIP-Token'] = token;
    if (body) headers['Content-Type'] = 'application/json';
    let res;
    try { res = await fetch(path, { method: body ? 'POST' : 'GET', headers, body: body ? JSON.stringify(body) : undefined }); }
    catch (e) { throw new ApiError('连不上 MoTrack，请确认程序还在运行', 0); }
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) { location.href = '/app/login.html'; throw new ApiError('请先登录', 401); }
    if (!res.ok) {
      const d = data.detail;
      const err = new ApiError(typeof d === 'string' ? d : (d && d.message) || '出了点问题，请稍后重试', res.status, d && d.code);
      if (res.status === 403 && err.code && err.code.startsWith('AI_')) { err.handled = true; accessDenied(err); }
      throw err;
    }
    return data;
  }

  let toastTimer = 0;
  function toast(text, bad) {
    const el = $('#toast');
    el.textContent = text; el.className = 'toast show' + (bad ? ' bad' : '');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.className = 'toast' + (bad ? ' bad' : ''); }, 3800);
  }
  /* ---------- busy + working overlay ---------- */
  function setBusy(v) {
    state.busy = v;
    $('#go').disabled = v; $('#send').disabled = v;
    $$('.vcard').forEach((b) => { b.disabled = v; });
    $('#go').textContent = v ? '剪辑中…' : (state.output ? '重新剪辑' : '开始剪辑');
  }
  // stage: a job stage key, 'CHAT' for an edit from the chat, or null to hide.
  function working(stage, pct) {
    const box = $('#working');
    box.classList.toggle('hidden', stage == null);
    if (stage == null) return;
    const flow = FLOWS[state.mode];
    const at = stage === 'CHAT' ? -1 : (flow.at[stage] != null ? flow.at[stage] : 0);
    $('#wsteps').innerHTML = stage === 'CHAT' ? '' : flow.steps.map((s, i) =>
      '<li class="' + (i < at ? 'done' : i === at ? 'now' : '') + '">' + (i < at ? '✓ ' : '') + esc(s) + '</li>').join('');
    $('#workingText').textContent = stage === 'CHAT'
      ? (state.mode === 'ai' ? 'AI 剪辑师正在按你说的改…' : '正在按你说的改…') : (STAGE_TEXT[stage] || '处理中…');
    $('#workingSub').textContent = stage === 'AI_LOOKING' ? '会跳过观众、广告和没看点的镜头' : '';
    const bar = $('#workingBar');
    if (typeof pct === 'number') { bar.classList.add('fixed'); bar.style.width = Math.max(4, pct) + '%'; }
    else { bar.classList.remove('fixed'); bar.style.width = ''; }
  }

  /* ---------- top steps ---------- */
  function renderSteps() {
    const now = state.busy ? 2 : state.output ? 3 : state.video ? 1 : 0;
    $$('#steps li').forEach((li, i) => { li.className = i < now ? 'done' : i === now ? 'now' : ''; });
  }

  /* ---------- videos ---------- */
  const UPLOAD_STAGE = { uploading: '上传中', TRANSCODING: '转换格式', ANALYZING: '找精彩镜头' };
  const etaSeen = {};
  function eta(v) {
    const key = v.id + ':' + (v.stage || v.status), p = Number(v.progress) || 0, now = Date.now();
    const s0 = etaSeen[key] || (etaSeen[key] = { t: now, p });
    const done = p - s0.p, secs = (now - s0.t) / 1000;
    if (secs < 8 || done < 2 || p >= 100) return '';
    const left = secs / done * (100 - p);
    return left < 50 ? ' · 快好了' : ' · 还要约 ' + Math.ceil(left / 60) + ' 分钟';
  }
  function renderVideos() {
    const upload = '<button class="vcard upload" id="uploadCard" type="button"><div class="thumb">＋</div>' +
      '<div><b>上传视频</b><small>' + (state.me && !state.me.admin ? '10 分钟以内的比赛' : '从相册选一段比赛') + '</small></div></button>';
    $('#videos').innerHTML = upload + state.videos.map((v) => {
      const del = v.upload ? '<span class="del" data-del="' + esc(v.id) + '" role="button" aria-label="删除" title="删除">×</span>' : '';
      if (v.status && v.status !== 'ready') {
        const failed = v.status === 'failed';
        const label = failed ? (v.error || '处理失败') : (UPLOAD_STAGE[v.stage] || UPLOAD_STAGE[v.status] || '处理中') + ' ' + (v.progress || 0) + '%' + eta(v);
        return '<div class="vcard pending' + (failed ? ' failed' : '') + '"><div class="thumb ph"></div><div><b>' + esc(v.name) + '</b><small>' + esc(label) + '</small>' +
          (failed ? '' : '<div class="bar"><i style="width:' + (v.progress || 0) + '%"></i></div>') + '</div>' + (failed ? del : '') + '</div>';
      }
      return '<button class="vcard' + (state.video && state.video.id === v.id ? ' active' : '') + '" data-id="' + esc(v.id) + '" type="button">' +
        '<div class="thumb ph">' + img(v.poster) + '<span class="dur">' + clock(v.duration) + '</span></div>' +
        '<div><b>' + esc(v.name) + '</b><small>' + esc(v.subtitle || '') + '</small></div>' + del + '</button>';
    }).join('');
  }
  let videoPoll = 0;
  async function loadVideos(selectId) {
    try { state.videos = await api('/api/v1/videos'); } catch (e) { toast(e.message, true); return; }
    const ready = state.videos.filter((v) => v.status === 'ready');
    if (state.video && !ready.some((v) => v.id === state.video.id)) state.video = null;
    renderVideos();
    const want = selectId && ready.find((v) => v.id === selectId);
    if (want) selectVideo(want.id); else if (!state.video && ready.length) selectVideo(ready[0].id);
    clearTimeout(videoPoll);
    if (state.videos.some((v) => v.status && !['ready', 'failed'].includes(v.status))) videoPoll = setTimeout(pollUploads, 1500);
  }
  async function pollUploads() {
    const before = new Set(state.videos.filter((v) => v.status === 'ready').map((v) => v.id));
    try { state.videos = await api('/api/v1/videos'); } catch (e) { return; }
    renderVideos();
    const fresh = state.videos.find((v) => v.status === 'ready' && v.upload && !before.has(v.id));
    if (fresh && !state.busy) { selectVideo(fresh.id); toast('「' + fresh.name + '」处理好了，可以开始剪了 🎉'); }
    if (state.videos.some((v) => v.status && !['ready', 'failed'].includes(v.status))) videoPoll = setTimeout(pollUploads, 1500);
  }
  async function uploadVideo(file) {
    if (!file) return;
    if (file.size > 1024 * 1024 * 1024) { toast('视频不能超过 1GB', true); return; }
    if (state.me && !state.me.admin) {
      const secs = await new Promise((done) => {
        const v = document.createElement('video'), url = URL.createObjectURL(file);
        const end = (d) => { URL.revokeObjectURL(url); done(d); };
        v.preload = 'metadata'; v.onloadedmetadata = () => end(v.duration); v.onerror = () => end(0);
        setTimeout(() => end(0), 4000); v.src = url;
      });
      if (secs > 600) { toast('视频有 ' + Math.round(secs / 60) + ' 分钟，请剪到 10 分钟以内再上传', true); $('#videoFile').value = ''; return; }
    }
    let uid;
    try {
      const started = await api('/api/v1/uploads', { name: file.name || 'video.mp4', size: file.size });
      uid = started.id;
      const chunk = started.chunk || 8 * 1024 * 1024;
      state.videos.unshift({ id: uid, name: file.name || '我的视频', upload: true, status: 'uploading', progress: 0 });
      renderVideos();
      let offset = 0, tries = 0;
      while (offset < file.size) {
        const piece = file.slice(offset, Math.min(file.size, offset + chunk));
        let res;
        try { res = await fetch('/api/v1/uploads/' + uid + '?offset=' + offset, { method: 'PUT', body: piece }); }
        catch (e) { res = null; }
        if (res && res.status === 409) { offset = (await res.json()).expected; continue; }  // resume where the server is
        if (!res || !res.ok) {
          if (++tries > 4) throw new ApiError('网络不稳定，上传中断了，请重试', 0);
          await new Promise((r) => setTimeout(r, 1500 * tries)); continue;
        }
        tries = 0; offset = (await res.json()).received;
        const v = state.videos.find((x) => x.id === uid);
        if (v) { v.progress = Math.round(100 * offset / file.size); renderVideos(); }
      }
      await api('/api/v1/uploads/' + uid + '/finish', {});
      toast('上传好了，正在转换格式、找精彩镜头…');
      loadVideos();
    } catch (e) {
      toast(e.message || '上传失败', true);
      loadVideos();
    }
    $('#videoFile').value = '';
  }
  async function deleteVideo(id) {
    const v = state.videos.find((x) => x.id === id);
    if (!v || !confirm('删除「' + v.name + '」？剪好的集锦也会一起删除。')) return;
    try {
      const res = await fetch('/api/v1/videos/' + encodeURIComponent(id), { method: 'DELETE' });
      if (!res.ok) throw new Error(((await res.json().catch(() => ({}))).detail) || '删除失败');
      if (state.video && state.video.id === id) state.video = null;
      toast('已删除'); loadVideos();
    } catch (e) { toast(e.message, true); }
  }
  function selectVideo(id) {
    const v = state.videos.find((x) => x.id === id);
    if (!v || state.busy) return;
    state.video = v;
    renderVideos();
    $('#source').src = v.source || ('/api/v1/videos/' + encodeURIComponent(v.id) + '/source');
    if (v.poster) $('#source').poster = v.poster;
    resetOutput(); resetChat(); renderDurations(); loadMoments(); renderSteps();
  }

  /* ---------- duration ---------- */
  function maxDuration() { return Math.max(5, Math.min(60, Math.floor((state.video && state.video.duration) || 60))); }
  function renderDurations() {
    const max = maxDuration();
    if (state.duration > max) state.duration = max;
    const list = DURATIONS.filter((d) => d < max);
    if (!list.includes(max)) list.push(max);
    $('#durations').innerHTML = list.map((d) =>
      '<button class="chip' + (d === state.duration ? ' active' : '') + '" data-d="' + d + '" type="button">' +
      (d === max && !DURATIONS.includes(d) ? '整段 ' : '') + d + ' 秒</button>').join('');
    $('#durationHint').textContent = max < 60 ? '这段原片只有 ' + clock(state.video.duration) + '，最长剪 ' + max + ' 秒' : '';
  }

  /* ---------- style ---------- */
  function segValue(id) { const b = $('#' + id + ' button.active'); return b ? b.dataset.value : ''; }
  function setSeg(id, value) {
    const buttons = $$('#' + id + ' button');
    const match = buttons.find((b) => b.dataset.value === value);
    if (match) buttons.forEach((b) => b.classList.toggle('active', b === match));
  }
  function controlsStyle() {
    return { pace: segValue('pace'), transition: segValue('transition'), look: segValue('look'),
             zoom: $('#zoom').checked, music: state.music, slow_motion: $('#slowmo').checked };
  }
  function editStyle() {
    const title = $('#title').value.trim();
    return { ...controlsStyle(), captions: $('#captions').checked, aspect: segValue('aspect'),
             title: title || (state.titleOff ? '' : null) };
  }
  function renderPresets() {
    $('#presets').innerHTML = PRESETS.map((p) =>
      '<button class="preset ' + p.cls + (state.preset === p.id ? ' active' : '') + '" data-id="' + p.id + '" type="button">' +
      '<span class="emo">' + p.emo + '</span><b>' + esc(p.name) + '</b><small>' + esc(p.desc) + '</small></button>').join('');
    $('#customNote').classList.toggle('hidden', state.preset !== 'custom');
  }
  function applyPreset(id) {
    const p = PRESETS.find((x) => x.id === id); if (!p) return;
    const s = p.style;
    setSeg('pace', s.pace); setSeg('transition', s.transition); setSeg('look', s.look);
    $('#zoom').checked = s.zoom; $('#slowmo').checked = s.slow_motion;
    state.music = s.music; state.preset = id;
    renderMusic(); renderPresets();
  }
  function matchPreset() {
    const cur = controlsStyle();
    const hit = PRESETS.find((p) => STYLE_KEYS.every((k) => p.style[k] === cur[k]));
    state.preset = hit ? hit.id : 'custom';
    renderPresets();
  }
  // The style the editor actually used; mirroring it keeps chat edits from being undone.
  function applyStyle(s) {
    if (!s || typeof s !== 'object') return;
    setSeg('pace', s.pace); setSeg('transition', s.transition); setSeg('look', s.look); setSeg('aspect', s.aspect);
    if (typeof s.captions === 'boolean') $('#captions').checked = s.captions;
    if (typeof s.slow_motion === 'boolean') $('#slowmo').checked = s.slow_motion;
    if (typeof s.zoom === 'boolean') $('#zoom').checked = s.zoom;
    if (typeof s.music === 'string') state.music = s.music;
    if ('title' in s) { state.titleOff = s.title === ''; $('#title').value = s.title || ''; }
    renderMusic(); matchPreset();
  }
  function styleText(s) {
    if (!s) return '';
    const preset = PRESETS.find((p) => STYLE_KEYS.every((k) => p.style[k] === s[k]));
    const bits = [preset ? preset.emo + ' ' + preset.name : '自定义风格'];
    if (s.aspect === '9:16') bits.push('竖屏');
    if (s.music && s.music !== 'none') bits.push('♪ ' + musicTitle(s.music));
    return bits.join(' · ');
  }
  /* ---------- music ---------- */
  function musicTitle(id) {
    const m = state.musicList.find((x) => x.id === id);
    return m ? m.title : (id && id.startsWith('upload:') ? '我的音乐' : '配乐');
  }
  function renderMusic() {
    const items = [{ id: 'none', title: '原声' }, ...state.musicList];
    $('#music').innerHTML = items.map((m) =>
      '<button class="chip' + (m.id === state.music ? ' active' : '') + '" data-id="' + esc(m.id) + '" type="button" title="' + esc(m.desc || m.title) + '">' +
      (m.id.startsWith('upload:') ? '🎵 ' : '') + esc(m.title) + '</button>').join('');
    $('#musicPreview').disabled = state.music === 'none';
    $('#musicHint').textContent = state.music === 'none' ? '内置配乐会自动卡点'
      : state.music.startsWith('upload:') ? '用你的音乐，原声调低' : '已开卡点：每一刀落在鼓点上';
  }
  async function loadMusic() {
    try {
      const data = await api('/api/v1/music');
      state.musicList = [...(data.presets || []), ...(data.uploads || [])];
    } catch (e) { state.musicList = []; }
    renderMusic();
  }
  function stopPreview() { const a = $('#musicAudio'); if (!a.paused) a.pause(); $('#musicPreview').textContent = '▶ 试听'; }
  function previewMusic() {
    const audio = $('#musicAudio');
    if (!audio.paused) { stopPreview(); return; }
    if (state.music === 'none') return;
    audio.src = '/api/v1/music/' + encodeURIComponent(state.music) + '/preview';
    audio.play().then(() => { $('#musicPreview').textContent = '⏸ 停止'; }).catch(() => toast('试听失败，请稍后重试', true));
  }
  async function uploadMusic(file) {
    if (!file) return;
    if (file.size > 30 * 1024 * 1024) { toast('音乐文件不能超过 30MB', true); return; }
    const form = new FormData(); form.append('file', file);
    toast('正在上传音乐…');
    try {
      const res = await fetch('/api/v1/music/upload', { method: 'POST', body: form });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : '上传失败');
      await loadMusic();
      state.music = data.id; renderMusic(); matchPreset();
      toast('已添加「' + data.title + '」，下次剪辑会用它做配乐');
    } catch (e) { toast(e.message, true); }
    $('#musicFile').value = '';
  }

  /* ---------- mode / VIP ---------- */
  function renderMode() {
    const ai = state.mode === 'ai';
    $$('.mode').forEach((b) => b.classList.toggle('active', b.dataset.mode === state.mode));
    $('#aiLock').classList.toggle('hidden', state.vip);
    $('#go').classList.toggle('vip', ai);
    $('#goHint').textContent = ai ? 'AI 先看画面再剪，约 30–60 秒' : '约 5 秒出片';
    $('#avatar').textContent = ai ? '👑' : '🤖';
    $('#avatar').classList.toggle('vip', ai);
    $('#assistantName').textContent = ai ? 'AI 剪辑师' : '剪辑助手';
    $('#assistantSub').textContent = ai ? '会看画面，随便说' : '一句话告诉我怎么改';
    renderSuggest();
    const pill = $('#vipPill');
    pill.classList.toggle('on', state.vip);
    const me = state.me;
    if (me && state.dev) {
      pill.classList.toggle('on', me.admin || me.status === 'approved');
      pill.innerHTML = me.admin ? '🛡<span class="long"> 管理员</span>' : me.status === 'approved' ? '✅<span class="long"> 已认证</span>'
        : me.status === 'pending' ? '⏳<span class="long"> 审核中</span>' : '✨ ' + (me.left || 0) + '<span class="long"> 次 AI</span>';
    } else {
      pill.innerHTML = state.dev ? '👑 VIP<span class="long"> · 开发模式</span>' : state.vip ? '👑 VIP<span class="long"> 会员</span>' : '👑 <span class="long">开通 </span>VIP';
    }
    renderQuota();
  }
  function renderQuota() {
    const q = $('#quota'), me = state.me;
    if (!me || me.admin || me.status === 'approved') { q.classList.add('hidden'); return; }
    q.classList.remove('hidden'); q.classList.toggle('done', me.status !== 'free' || !me.left);
    q.innerHTML = me.status === 'pending' ? '⏳ 申请已提交，管理员审核通过后可继续使用 AI 功能'
      : me.status === 'rejected' ? 'AI 功能申请没有通过，基础剪辑仍可免费使用'
      : me.left ? 'AI 功能可免费体验 <b>' + me.left + '</b> 次，之后留下姓名审核即可继续使用'
      : 'AI 免费体验已用完 <button class="link" id="quotaApply" type="button">留下姓名继续使用 →</button>';
  }
  let mePoll = 0;
  async function loadMe() {
    const before = state.me;
    try { state.me = await api('/api/v1/me'); } catch (e) { return; }
    $('#adminBtn').classList.toggle('hidden', !state.me.admin);
    const n = state.me.pending || 0;
    $('#adminBadge').textContent = n; $('#adminBadge').classList.toggle('hidden', !n);
    if (state.me.admin) { clearTimeout(mePoll); mePoll = setTimeout(loadMe, 60000); }
    const left = state.me.left;
    if (before && before.left != null && left != null && left < before.left && left > 0 && left <= 2)
      toast('AI 免费次数还剩 ' + left + ' 次，用完后留下姓名就能继续用');
    renderMode();
    if (state.me.status === 'pending') pollApproval();
  }
  let approvalTimer = 0;
  function pollApproval() {
    clearTimeout(approvalTimer);
    approvalTimer = setTimeout(async () => {
      const before = state.me && state.me.status;
      await loadMe();
      if (before === 'pending' && state.me && state.me.status === 'approved') {
        $('#accessDialog').classList.add('hidden');
        toast('🎉 审核通过了，AI 功能可以继续使用');
      }
    }, 20000);
  }
  function openAccess(message) {
    const st = state.me ? state.me.status : 'free';
    $('#accessApply').classList.toggle('hidden', st === 'pending' || st === 'rejected');
    $('#accessPending').classList.toggle('hidden', st !== 'pending');
    $('#accessRejected').classList.toggle('hidden', st !== 'rejected');
    if (message && st === 'free') $('#accessSub').textContent = message + '。基础剪辑一直免费。';
    $('#accessPendingText').textContent = '已收到「' + ((state.me && state.me.name) || '') + '」的申请，管理员审核通过后就能继续使用 AI 功能。';
    $('#accessDialog').classList.remove('hidden');
    if (st === 'free') setTimeout(() => $('#accessName').focus(), 50);
  }
  function accessDenied(err) {
    if (state.me) state.me.status = err.code === 'AI_PENDING' ? 'pending' : err.code === 'AI_REJECTED' ? 'rejected' : state.me.status;
    if (state.me && err.code === 'AI_LOGIN_REQUIRED') state.me.left = 0;
    renderMode(); openAccess(err.message.replace(/[。.]$/, ''));
  }
  async function applyAccess() {
    const name = $('#accessName').value.trim();
    if (name.length < 2) { toast('请填写姓名（至少 2 个字）', true); $('#accessName').focus(); return; }
    $('#accessSend').disabled = true;
    try {
      state.me = Object.assign(state.me || {}, await api('/api/v1/access/apply', { name, note: $('#accessNote').value.trim() }));
      renderMode(); openAccess();
      if (state.me.status === 'pending') pollApproval();
    } catch (e) { toast(e.message, true); }
    finally { $('#accessSend').disabled = false; }
  }
  function setMode(mode) {
    if (mode === state.mode || state.busy) return;
    if (mode === 'ai' && !state.vip) { openVip(); return; }
    state.mode = mode;
    renderMode(); resetChat(); loadMoments();
  }
  function openVip() {
    if (state.dev && state.me && !state.me.admin) { openAccess(); return; }
    if (state.dev && state.me && state.me.admin) { location.href = '/admin'; return; }
    if (state.vip) { toast(state.dev ? '开发模式下所有人都是 VIP，直接用 AI 剪辑师吧' : '你已经是 VIP 会员'); return; }
    $('#vipDialog').classList.remove('hidden');
    setTimeout(() => $('#vipCode').focus(), 50);
  }
  function closeVip() { $('#vipDialog').classList.add('hidden'); }
  async function activate() {
    const code = $('#vipCode').value.trim();
    if (!code) { $('#vipCode').focus(); return; }
    $('#vipActivate').disabled = true;
    try {
      const r = await api('/api/v1/vip/activate', { code });
      store.set(r.token); state.vip = true;
      closeVip(); $('#vipCode').value = '';
      state.mode = 'ai'; renderMode(); resetChat(); loadMoments();
      toast('🎉 VIP 已开通，AI 网球剪辑师已就绪');
    } catch (e) { toast(e.message, true); }
    finally { $('#vipActivate').disabled = false; }
  }
  function lostVip() {
    store.set(''); state.vip = false;
    if (state.mode === 'ai') { state.mode = 'lightweight'; resetChat(); }
    renderMode();
  }
  async function checkVip() {
    try { const r = await api('/api/v1/vip/status'); state.vip = !!r.vip; state.dev = !!r.dev; } catch (e) { return; }
    if (!state.vip && store.get()) store.set('');
    if (state.vip) state.mode = 'ai';
  }
  async function checkBackend() {
    try { if (((await api('/api/v1/health')).api || 0) < API_VERSION) $('#staleBanner').classList.remove('hidden'); }
    catch (e) { /* the videos call reports a dead backend */ }
  }

  /* ---------- panes ---------- */
  function showPane(name) {
    state.pane = name;
    $$('.tabs button').forEach((b) => b.classList.toggle('active', b.dataset.pane === name));
    $$('.pane').forEach((p) => p.classList.toggle('active', p.id === 'pane-' + name));
    document.body.classList.toggle('chatting', name === 'chat');
    if (name === 'chat') { $('#chatDot').classList.add('hidden'); scrollLog(); }
  }
  function chatVisible() { return state.pane === 'chat' || window.matchMedia('(min-width: 1360px)').matches; }
  /* ---------- moments ---------- */
  async function loadMoments() {
    const video = state.video; if (!video) return;
    try {
      const data = await api('/api/v1/videos/' + video.id + '/events?mode=' + state.mode);
      if (video !== state.video) return;
      state.events = Array.isArray(data.events) ? data.events : [];
      state.shooting = data.shooting || null;
    } catch (e) { state.events = []; state.shooting = null; }
    renderMoments();
  }
  function momentScore(e) {
    const raw = Number(e.confidence != null ? e.confidence : (e.final_ranking_score != null ? e.final_ranking_score : e.raw_model_score)) || 0;
    return e.ai ? 0.35 * raw + 0.65 * (e.ai.highlight / 5) : raw;
  }
  function renderMoments() {
    const pool = state.events.filter((e) => !(e.ai && e.ai.tennis === false));
    const top = pool.slice().sort((a, b) => momentScore(b) - momentScore(a)).slice(0, 10)
      .sort((a, b) => Number(a.start_time) - Number(b.start_time));
    const reviewed = state.events.filter((e) => e.ai).length;
    $('#momentsNote').textContent = !top.length ? '点一下在原片里看'
      : (reviewed ? 'AI 看过 ' + reviewed + ' 个 · ' : '') + '共 ' + pool.length + ' 个，点一下在原片里看';
    const sh = state.shooting;
    const stats = $('#shootStats');
    if (sh && sh.attempts) {
      stats.innerHTML = '<span><b>' + sh.attempts + '</b>次出手</span><span class="made"><b>' + sh.made + '</b>进球</span>' +
        '<span><b>' + sh.missed + '</b>没进</span>' + (sh.fg_pct != null ? '<span><b>' + sh.fg_pct + '%</b>命中率</span>' : '') +
        (sh.unknown ? '<small>另有 ' + sh.unknown + ' 次篮筐不在画面里，看不出进没进</small>' : '');
      stats.classList.remove('hidden'); countUp(stats);
    } else stats.classList.add('hidden');
    if (!top.length) {
      $('#moments').innerHTML = '<div class="none">第一次剪辑时会自动找出精彩瞬间。</div>';
      return;
    }
    $('#moments').innerHTML = top.map((e) => {
      const mid = (Number(e.start_time) + Number(e.end_time || e.start_time)) / 2;
      const s = momentScore(e);
      const stars = s >= 0.6 ? '★★★' : s >= 0.35 ? '★★☆' : '★☆☆';
      const name = (e.display_name || '精彩动作').replace(/ · (进了|没进)$/, '');
      return '<button class="moment" data-t="' + Number(e.start_time) + '" type="button">' +
        '<div class="pic ph">' + img(frameUrl(mid)) + '<span class="time">' + clock(e.start_time) + '</span>' +
        (e.ai ? '<span class="ai-tag">AI 看过</span>' : '') +
        (e.outcome === 'made' ? '<span class="out-tag made">进球</span>' : e.outcome === 'missed' ? '<span class="out-tag">没进</span>' : '') + '</div>' +
        '<div class="info"><b>' + esc(name) + '</b><span class="stars">' + stars + '</span>' +
        (e.ai && e.ai.note ? '<span class="note">' + esc(e.ai.note) + '</span>' : '') + '</div></button>';
    }).join('');
  }

  /* ---------- review (复盘) ---------- */
  function seekSource(t) {
    if (!state.output) $('#viewToggle').classList.remove('hidden');
    showView('source'); $('#empty').classList.add('hidden');
    const v = $('#source'); v.currentTime = Math.max(0, Number(t) - 1); v.play().catch(() => {});
    $('.player').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
  const OUT_ZH = { made: '进了', missed: '没进', unknown: '看不出' };
  async function openReview() {
    const video = state.video; if (!video) { toast('先选一场比赛'); return; }
    $('#reviewSub').textContent = video.name;
    $('#reviewBody').innerHTML = '<div class="spin"></div>';
    $('#reviewDialog').classList.remove('hidden');
    try {
      const base = '/api/v1/videos/' + encodeURIComponent(video.id);
      const [r, pl] = await Promise.all([api(base + '/review'), api(base + '/players').catch(() => ({ players: [] }))]);
      if (video !== state.video) return;
      r.players = pl.players || []; r.teams = !!pl.teams;
      state.review = r; renderReview(r); countUp($('#reviewBody'));
    } catch (e) {
      $('#reviewBody').innerHTML = '<p class="sub">' + esc(e.message) + '</p>';
    }
  }
  function renderReview(r) {
    if (r.kind === 'tennis_review') {
      const moments = r.moments.map((m) => '<button class="rv-shot" type="button" data-t="' + m.time + '"><time>' + esc(m.clock) + '</time>' + esc(m.label) + '<b>回看</b></button>').join('');
      const periods = r.active_periods.map((p) => '<li><b>' + esc(p.label) + '</b> · ' + p.count + ' 个候选</li>').join('');
      const best = (r.best || []).map((m) => '<button class="rv-best" type="button" data-t="' + m.time + '"><span class="info"><b>🏆 ' + esc(m.clock + ' ' + m.label) + '</b><small>' + esc(m.reason) + '</small></span></button>').join('');
      const periodRows = (r.periods || []).filter((p) => p.count).map((p) => '<tr><td>' + esc(p.label) + '</td><td>' + p.count + '</td><td>' + (p.avg_score != null ? p.avg_score : '—') + '</td></tr>').join('');
      const recap = r.recap ? '<div class="rv-recap">' + esc(r.recap) + '</div>' : '<div class="rv-recap"><p class="sub" style="margin:0 0 10px">AI 会基于候选时段写一段教练视角的网球复盘，不会编造比分或胜负。</p><button class="go vip" id="recapGo" type="button">👑 生成 AI 复盘</button></div>';
      $('#reviewBody').innerHTML = '<div class="rv-tiles"><div class="rv-tile"><b>' + r.candidate_count + '</b><span>回看候选</span></div><div class="rv-tile"><b>' + esc(clock(r.duration)) + '</b><span>视频时长</span></div></div>' +
        (best ? '<section class="rv-bestbox"><div class="rv-besthead"><div><h4>🏆 优先回看</h4><p>按当前候选强度排序</p></div></div><div class="rv-bests">' + best + '</div></section>' : '') +
        '<div class="rv-h">重点时段</div><ul class="rv-notes">' + periods + '</ul>' +
        '<div class="rv-h">候选时间线 · 点一个点看原片</div><div class="rv-shots">' + moments + '</div>' +
        '<div class="rv-h">复盘建议</div><ul class="rv-notes">' + r.notes.map((n) => '<li>' + esc(n) + '</li>').join('') + '</ul>' +
        '<div class="rv-h">AI 复盘 <i class="tag vip">VIP</i></div>' + recap +
        (periodRows ? '<div class="rv-h">分时段候选</div><table class="rv-table"><tr><th>时段</th><th>候选</th><th>平均强度</th></tr>' + periodRows + '</table>' : '') +
        '<div class="rv-foot"><p class="fine">' + esc(r.caveat) + '</p><a class="btn" href="/api/v1/videos/' + encodeURIComponent(r.video_id) + '/review/export" download>⬇ 导出复盘报告</a></div>';
      return;
    }
    const s = r.stats, dur = Math.max(1, r.duration);
    const tile = (v, k, cls) => '<div class="rv-tile ' + (cls || '') + '"><b>' + esc(String(v)) + '</b><span>' + k + '</span></div>';
    const dots = r.shots.map((x) => '<button class="rv-dot ' + x.outcome + '" type="button" data-t="' + x.time + '" style="left:calc(10px + (100% - 20px) * ' +
      (x.time / dur).toFixed(4) + ')" title="' + esc(x.clock + ' ' + x.label) + '" aria-label="' + esc(x.clock + ' ' + x.label) + '"></button>').join('');
    const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => '<span>' + clock(dur * f) + '</span>').join('');
    const periods = r.periods.map((p) => '<tr><td>' + esc(p.label) + '</td><td>' + p.attempts + '</td><td class="' + (p.made ? 'made-txt' : '') + '">' + p.made +
      '</td><td>' + p.missed + '</td><td>' + (p.fg_pct != null ? p.fg_pct + '%' : '—') + '</td></tr>').join('');
    const shots = r.shots.map((x) => '<button class="rv-shot ' + x.outcome + '" type="button" data-t="' + x.time + '"><time>' + esc(x.clock) + '</time>' +
      esc(x.label.replace(/ · .*$/, '')) + '<b>' + OUT_ZH[x.outcome] + '</b></button>').join('');
    const TEAM = ['A 队', 'B 队'];
    const players = (r.players || []).map((p) => '<button class="rv-player" type="button" data-player="' + p.number + '">' +
      '<img loading="lazy" alt="" src="/api/v1/videos/' + encodeURIComponent(r.video_id) + '/players/' + p.id + '/photo">' +
      '<span><b>' + p.number + ' 号</b>' + (r.teams && p.team != null ? '<i>' + TEAM[p.team] + '</i>' : '') +
      '<small>' + p.attempts + ' 次出手 · 进 ' + p.made + (p.fg_pct != null ? ' · ' + p.fg_pct + '%' : '') + '</small></span></button>').join('');
    const MEDAL = ['🥇', '🥈', '🥉'];
    const bestList = r.best || [];
    const best = bestList.map((b) => '<button class="rv-best' + (b.rank === 1 ? ' first' : '') + '" type="button" data-t="' + b.time + '">' +
      '<span class="pic"><img loading="lazy" alt="" src="/api/v1/videos/' + encodeURIComponent(r.video_id) + '/frame?t=' + b.peak + '">' +
      '<i class="rank">' + (MEDAL[b.rank - 1] || b.rank) + '</i><i class="play">▶</i><time>' + esc(b.clock) + '</time></span>' +
      '<span class="info"><b>' + esc(b.label) + '</b><small>' + esc(b.reason) + '</small></span></button>').join('');
    const bestBlock = bestList.length ? '<section class="rv-bestbox"><div class="rv-besthead"><div><h4>🏆 本场最佳</h4>' +
      '<p>按现场欢呼、出手距离和连续得分自动排出</p></div><button class="go rv-bestgo" id="bestGo" type="button" data-n="' + bestList.length + '">🎬 一键生成最佳 ' +
      bestList.length + ' 球集锦</button></div><div class="rv-bests" >' + best + '</div></section>' : '';
    const notes = r.notes.concat(r.streaks.map((x) => x.text)).map((n) => '<li>' + esc(n) + '</li>').join('');
    const recap = r.recap ? '<div class="rv-recap">' + esc(r.recap) + '</div>'
      : '<div class="rv-recap"><p class="sub" style="margin:0 0 10px">AI 根据这些数据写一段教练视角的复盘，指出关键时段并给出建议。</p>' +
        '<button class="go vip" id="recapGo" type="button">👑 生成 AI 复盘</button></div>';
    $('#reviewBody').innerHTML =
      '<div class="rv-tiles">' + tile(s.attempts, '出手') + tile(s.made, '进球', 'made') + tile(s.missed, '没进') + tile(s.fg_pct != null ? s.fg_pct + '%' : '—', '命中率', 'pct') + '</div>' +
      bestBlock +
      '<div class="rv-h">出手时间线 · 点一个点看原片</div>' +
      (r.shots.length ? '<div class="rv-line"><div class="axis"></div>' + dots + '</div><div class="rv-ticks">' + ticks + '</div>' +
        '<div class="rv-legend"><span><i class="m"></i>进了</span><span><i class="x"></i>没进</span><span><i></i>看不出</span></div>'
        : '<p class="sub">这段视频没有检测到投篮出手。下面是其他值得回看的时刻：</p><div class="rv-shots">' +
          r.moments.map((m) => '<button class="rv-shot" type="button" data-t="' + m.time + '"><time>' + esc(m.clock) + '</time>' + esc(m.label) + '</button>').join('') + '</div>') +
      (players ? '<div class="rv-h">球员 · 点一个人生成他的个人集锦</div><div class="rv-players" data-scroller>' + players + '</div>' +
        '<p class="fine rv-pnote">号码按出手次数自动编排，不是球衣号。</p>' : '') +
      '<div class="rv-h">要点</div><ul class="rv-notes">' + notes + '</ul>' +
      '<div class="rv-h">AI 复盘 <i class="tag vip">VIP</i></div>' + recap +
      (r.shots.length ? '<div class="rv-h">分时段</div><table class="rv-table"><tr><th>时段</th><th>出手</th><th>进</th><th>没进</th><th>命中率</th></tr>' + periods + '</table>' +
        '<div class="rv-h">每次出手</div><div class="rv-shots">' + shots + '</div>' : '') +
      '<div class="rv-foot"><p class="fine">' + esc(r.caveat) + '</p><a class="btn" id="reviewExport" href="/api/v1/videos/' +
        encodeURIComponent(r.video_id) + '/review/export" download>⬇ 导出复盘报告</a></div>';
  }
  async function makeRecap() {
    if (!state.vip) { openVip(); return; }
    const r = state.review, btn = $('#recapGo'); if (!r || !btn) return;
    btn.disabled = true; btn.textContent = 'AI 正在写复盘…';
    try {
      const out = await api('/api/v1/videos/' + encodeURIComponent(r.video_id) + '/review/recap', {});
      r.recap = out.recap; if (state.review === r) renderReview(r);
    } catch (e) { if (!e.handled) toast(e.message, true); btn.disabled = false; btn.textContent = '👑 生成 AI 复盘'; }
    finally { loadMe(); }
  }

  /* ---------- horizontal scrollers ---------- */
  // Wrap each [data-scroller] row: visible scrollbar, edge fades, arrow buttons
  // on mouse devices, and the mouse wheel scrolls it sideways.
  function enhanceScroller(sc) {
    if (sc.dataset.enhanced) return; sc.dataset.enhanced = '1';
    const wrap = document.createElement('div'); wrap.className = 'hscroll';
    sc.parentNode.insertBefore(wrap, sc); wrap.appendChild(sc); sc.classList.add('sc');
    const mk = (cls, txt, dir) => {
      const b = document.createElement('button'); b.type = 'button'; b.className = 'hs-btn ' + cls; b.textContent = txt;
      b.setAttribute('aria-label', dir < 0 ? '向左' : '向右');
      b.addEventListener('click', () => sc.scrollBy({ left: dir * Math.max(160, sc.clientWidth * 0.8), behavior: 'smooth' }));
      wrap.appendChild(b);
    };
    mk('l', '‹', -1); mk('r', '›', 1);
    if (sc.dataset.hint) { const h = document.createElement('p'); h.className = 'hs-hint'; h.textContent = sc.dataset.hint + ' →'; wrap.after(h); }
    const sync = () => {
      const max = sc.scrollWidth - sc.clientWidth;
      wrap.classList.toggle('can-l', sc.scrollLeft > 4);
      wrap.classList.toggle('can-r', max - sc.scrollLeft > 4);
    };
    sc.addEventListener('scroll', sync, { passive: true });
    sc.addEventListener('wheel', (e) => {
      const max = sc.scrollWidth - sc.clientWidth;
      if (max <= 0 || Math.abs(e.deltaY) <= Math.abs(e.deltaX)) return;
      const next = Math.max(0, Math.min(max, sc.scrollLeft + e.deltaY));
      if (next === sc.scrollLeft) return;   // at the end: let the page scroll
      e.preventDefault(); sc.scrollLeft = next;
    }, { passive: false });
    if (window.ResizeObserver) new ResizeObserver(sync).observe(sc);
    new MutationObserver(() => requestAnimationFrame(sync)).observe(sc, { childList: true });
    window.addEventListener('resize', sync);
    sync();
  }
  $$('[data-scroller]').forEach(enhanceScroller);

  /* ---------- motion helpers ---------- */
  var calm = window.matchMedia('(prefers-reduced-motion: reduce)');
  // Lists redrawn on every click or upload poll animate in only the first time.
  ['#videos', '#presets', '#durations'].forEach((sel) => {
    const el = $(sel); if (!el) return;
    const watch = new MutationObserver(() => { watch.disconnect(); settle(); });
    function settle() {
      if (el.children.length) setTimeout(() => el.classList.add('settled'), 900);
      else watch.observe(el, { childList: true });
    }
    settle();
  });
  // Dialogs: when code hides one, play the close animation first.
  $$('.overlay').forEach((ov) => {
    let busy = false;
    new MutationObserver(() => {
      const hidden = ov.classList.contains('hidden');
      if (!hidden) { if (!ov.classList.contains('closing')) ov.dataset.shown = '1'; return; }
      if (busy || calm.matches || ov.dataset.shown !== '1') return;
      busy = true; ov.dataset.shown = '';
      ov.classList.remove('hidden'); ov.classList.add('closing');
      setTimeout(() => { ov.classList.remove('closing'); ov.classList.add('hidden'); busy = false; }, 190);
    }).observe(ov, { attributes: true, attributeFilter: ['class'] });
  });
  // Numbers in stat tiles count up from zero.
  function countUp(root) {
    if ((calm && calm.matches) || !root) return;
    root.querySelectorAll('.rv-tile b, .shoot-stats b').forEach((b) => {
      const m = /^(\d+)(%?)$/.exec(b.textContent.trim()); if (!m) return;
      const end = Number(m[1]), unit = m[2], t0 = performance.now();
      if (!end) return;
      b.dataset.count = '1';
      const step = (now) => {
        const k = Math.min(1, (now - t0) / 700);
        b.textContent = Math.round(end * (1 - Math.pow(1 - k, 3))) + unit;
        if (k < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    });
  }

  /* ---------- first-visit guide ---------- */
  const GUIDE_KEY = 'motrack_guide_v1';
  function guideSteps() {
    const ai = state.me && !state.me.admin && state.me.status !== 'approved'
      ? '切到「AI 剪辑师」会先看画面再挑镜头，可以免费体验 ' + (state.me.free || 6) + ' 次，之后留下姓名，审核通过就能继续用。'
      : '切到「AI 剪辑师」，它会先看画面再挑镜头，还会写一段解说。';
    return [
      { el: '#videos', pane: 'edit', title: '先选一场比赛', text: '点一段示例比赛就能直接试。想剪自己的比赛，点「＋ 上传视频」，从手机相册或电脑里选。' },
      { el: '#presets', pane: 'edit', title: '选风格和时长', text: '一个风格同时决定配乐、转场和调色。下面再选集锦要多长，竖屏适合发抖音、视频号。' },
      { el: '#go', pane: 'edit', title: '点「开始剪辑」', text: '基础剪辑几秒出片，全程免费。' + ai },
      { el: '.moments-card', title: '查看精彩瞬间', text: '这里列出自动识别到的发球、击球和精彩镜头；点一下可以跳到原片查看。' },
      { el: () => (getComputedStyle($('.tabs')).display !== 'none' ? $('.tabs [data-pane=chat]') : $('#pane-chat')), title: '一句话修改',
        text: '剪好后在「对话」里直接说，比如“去掉第 2 段”“改成竖屏”“节奏快一点”。随时点右上角「?」可以再看这份指南。' }
    ];
  }
  let guideAt = 0, guideList = [];
  function openGuide() {
    guideList = guideSteps(); guideAt = 0;
    $('#guide').classList.remove('hidden'); showGuideStep();
  }
  function closeGuide() {
    $('#guide').classList.add('hidden');
    try { localStorage.setItem(GUIDE_KEY, '1'); } catch (e) { /* private mode */ }
    if (narrow()) window.scrollTo({ top: 0, behavior: 'smooth' });
  }
  function showGuideStep() {
    const step = guideList[guideAt];
    if (step.pane && state.pane !== step.pane) showPane(step.pane);
    $('#guideStep').textContent = '第 ' + (guideAt + 1) + ' 步，共 ' + guideList.length + ' 步';
    $('#guideTitle').textContent = step.title; $('#guideText').textContent = step.text;
    $('#guideDots').innerHTML = guideList.map((_, i) => '<i class="' + (i === guideAt ? 'on' : '') + '"></i>').join('');
    $('#guideNext').textContent = guideAt === guideList.length - 1 ? '开始使用' : '下一步';
    const el = typeof step.el === 'function' ? step.el() : $(step.el);
    if (el) el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    setTimeout(placeGuide, 350);
  }
  function placeGuide() {
    if ($('#guide').classList.contains('hidden')) return;
    const step = guideList[guideAt], el = typeof step.el === 'function' ? step.el() : $(step.el);
    const hole = $('#guideHole'), card = $('#guideCard');
    const vw = window.innerWidth, vh = window.innerHeight, cw = card.offsetWidth, ch = card.offsetHeight;
    const r = el && el.getBoundingClientRect();
    if (!r || !r.width || r.bottom < 0 || r.top > vh) {
      hole.className = 'guide-hole center'; hole.removeAttribute('style');
      card.style.left = (vw - cw) / 2 + 'px'; card.style.top = Math.max(16, (vh - ch) / 2) + 'px'; return;
    }
    hole.className = 'guide-hole';
    const pad = 6, top = Math.max(4, r.top - pad), bottom = Math.min(vh - 4, r.bottom + pad);
    Object.assign(hole.style, { left: Math.max(4, r.left - pad) + 'px', top: top + 'px', width: Math.min(vw - 8, r.width + 2 * pad) + 'px', height: (bottom - top) + 'px' });
    const below = vh - bottom - 12, above = top - 12;
    if (below < ch && above < ch && Math.max(r.left, vw - r.right) >= cw + 28) {
      // tall target (e.g. the chat column): put the card beside it
      card.style.left = (r.left >= vw - r.right ? r.left - cw - 14 : r.right + 14) + 'px';
      card.style.top = Math.min(Math.max(12, r.top), vh - ch - 12) + 'px';
      return;
    }
    let y = below >= ch ? bottom + 12 : above >= ch ? top - ch - 12 : (below >= above ? vh - ch - 12 : 12);
    if (below < ch && above < ch) {
      // no room either side: shrink the highlight to the part the card leaves visible
      const keepTop = below >= above;
      hole.style.top = (keepTop ? top : y + ch + 10) + 'px';
      hole.style.height = Math.max(40, (keepTop ? y - 10 : bottom) - (keepTop ? top : y + ch + 10)) + 'px';
    }
    card.style.top = Math.max(12, Math.min(y, vh - ch - 12)) + 'px';
    card.style.left = Math.min(Math.max(16, r.left), vw - cw - 16) + 'px';
  }
  function maybeGuide() {
    let seen = false;
    try { seen = !!localStorage.getItem(GUIDE_KEY); } catch (e) { seen = true; }
    if (!seen) setTimeout(openGuide, 600);
  }

  /* ---------- output ---------- */
  function showView(view) {
    $$('#viewToggle button').forEach((b) => b.classList.toggle('active', b.dataset.view === view));
    const hl = view === 'highlight';
    $('#highlight').classList.toggle('hidden', !hl); $('#source').classList.toggle('hidden', hl);
    (hl ? $('#source') : $('#highlight')).pause();
    $('#empty').classList.toggle('hidden', !hl || !!state.output);
  }
  function resetOutput() {
    state.output = null;
    $('#highlight').removeAttribute('src'); $('#highlight').load();
    ['#result', '#commentary', '#film', '#viewToggle'].forEach((s) => $(s).classList.add('hidden'));
    showView('highlight'); setBusy(state.busy); renderSteps();
  }
  function adoptOutput(out) {
    if (!out || !out.path) return;
    state.output = out;
    const url = artifactUrl(out.path, true);
    $('#highlight').src = url; $('#download').href = url + '&dl=1';
    prepareShare(url);
    $('#viewToggle').classList.remove('hidden');
    showView('highlight');
    $('#highlight').play().catch(() => {});
    const segs = Array.isArray(out.segments) ? out.segments : [];
    $('#resultBadge').textContent = (Number(out.duration) || 0).toFixed(1) + ' 秒 · ' + segs.filter((s) => !s.replay).length + ' 段';
    $('#resultStyle').textContent = styleText(out.edit_style);
    $('#result').classList.remove('hidden');
    const c = (out.commentary || '').trim();
    $('#commentary').classList.toggle('hidden', !c); $('#commentaryText').textContent = c;
    renderFilm(segs);
    if (!chatVisible()) $('#chatDot').classList.remove('hidden');
    setBusy(state.busy); renderSteps();
    if (narrow()) $('.player').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  function renderFilm(segs) {
    $('#film').classList.toggle('hidden', !segs.length);
    if (!segs.length) return;
    $('#filmMeta').textContent = segs.some((s) => s.replay) ? '含慢动作回放' : '';
    let n = 0;
    $('#filmTrack').innerHTML = segs.map((s) => {
      const label = s.replay ? '慢动作' : '第 ' + (++n) + ' 段';
      const at = Number(s.source_start) + Math.min(1, (Number(s.source_end) - Number(s.source_start)) / 2);
      return '<button class="seg-item ph" data-out="' + Number(s.output_start) + '"' + (s.replay ? '' : ' data-n="' + n + '"') + ' type="button" ' +
        'title="' + esc((s.label || '') + ' · 原片 ' + clock(s.source_start)) + '" ' +
        'style="flex-grow:' + Math.max(0.5, Number(s.output_duration) || 1) + '">' + img(frameUrl(at)) +
        '<span>' + label + '</span></button>';
    }).join('');
    $$('#filmTrack .seg-item').forEach((b, i) => b.classList.toggle('replay', !!segs[i].replay));
  }
  /* ---------- generate ---------- */
  async function pollJob(id) {
    for (let i = 0; i < 600; i++) {  // up to ~8 minutes for a first-time scan
      const job = await api('/api/v1/jobs/' + id);
      if (job.status === 'completed') return job;
      if (job.status === 'failed') throw new ApiError((job.error && job.error.message) || '这次没剪成，请稍后重试', 500);
      working(job.stage, job.stage === 'ANALYZING' && typeof job.progress === 'number' ? job.progress : null);
      await new Promise((r) => setTimeout(r, 700));
    }
    throw new ApiError('等待时间太长了，请稍后重试', 504);
  }
  const PREFS_KEY = 'motrack_prefs';
  function savePrefs() {
    try { localStorage.setItem(PREFS_KEY, JSON.stringify({ preset: state.preset, duration: state.duration, aspect: segValue('aspect'), focus: segValue('focus') })); }
    catch (e) { /* private mode */ }
  }
  function restorePrefs() {
    let p = null;
    try { p = JSON.parse(localStorage.getItem(PREFS_KEY) || 'null'); } catch (e) { return; }
    if (!p) return;
    if (PRESETS.some((x) => x.id === p.preset)) applyPreset(p.preset);
    if (DURATIONS.includes(p.duration)) state.duration = p.duration;
    if (['16:9', '9:16'].includes(p.aspect)) setSeg('aspect', p.aspect);
    if (['serve', 'rally', 'all'].includes(p.focus)) setSeg('focus', p.focus);
  }
  async function generate() {
    if (state.busy || !state.video) return;
    stopPreview(); savePrefs(); setBusy(true); renderSteps(); working('QUEUED');
    if (narrow()) $('.player').scrollIntoView({ behavior: 'smooth', block: 'start' });
    try {
      const shots = segValue('focus') !== 'all';
      const submitted = await api('/api/v1/highlights', {
        video_id: state.video.id, mode: state.mode, target_duration: state.duration,
        event_types: segValue('focus') === 'serve' ? ['serve', 'tennis_hit'] : shots ? ['tennis_hit'] : ['tennis_action_candidate'],
        edit_style: editStyle()
      });
      const job = await pollJob(submitted.id);
      const result = job.result || {};
      state.sessionId = result.session_id || submitted.session_id || state.sessionId;
      applyStyle(result.edit_style);
      working(null);
      adoptOutput(result.output);
      const out = result.output || {};
      addMsg('agent', state.mode === 'ai'
        ? '集锦剪好了，解说在视频下面。想改就直接说，比如“把没看点的去掉”“改成竖屏”。'
        : '剪好了：' + (Number(out.duration) || 0).toFixed(1) + ' 秒。想调整可以说“缩短到 10 秒”“换一首”。');
      toast(out.duration_note || '集锦剪好了 🎉 可以播放、下载，或者在对话里继续改');
      loadMoments();
    } catch (e) {
      if (e.code === 'VIP_REQUIRED') { lostVip(); openVip(); } else if (!e.handled) toast(e.message, true);
    } finally {
      working(null); setBusy(false); renderSteps(); if (state.mode === 'ai') loadMe();
    }
  }

  /* ---------- chat ---------- */
  function scrollLog() { const b = $('#pane-chat .pane-body'); b.scrollTop = b.scrollHeight; }
  function addMsg(role, text, extra) {
    const el = document.createElement('div');
    el.className = 'msg ' + role;
    el.innerHTML = esc(text).replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>') + (extra || '');
    $('#log').appendChild(el); scrollLog();
    return el;
  }
  function renderSuggest(list) {
    const items = Array.isArray(list) && list.length ? list : SUGGEST[state.mode];
    $('#suggest').innerHTML = items.map((t) => '<button type="button">' + esc(t) + '</button>').join('');
  }
  function selectSegment(n) {
    state.selected = n || null;
    $$('#filmTrack .seg-item').forEach((b) => b.classList.toggle('picked', !!n && Number(b.dataset.n) === n));
    $('#selChip').classList.toggle('hidden', !n);
    if (n) $('#selText').textContent = '已选第 ' + n + ' 段，可以说“去掉这段”“这段字幕改成…”';
  }
  function resetChat() {
    state.sessionId = ''; selectSegment(null); renderSuggest();
    $('#log').innerHTML = '';
    addMsg('agent', state.mode === 'ai'
      ? '嗨，我是你的 AI 剪辑师 👑 我会先看画面再挑镜头。\n先点“开始剪辑”，或者直接告诉我想要什么，比如“来个 20 秒竖屏网球集锦”。'
      : '你好！先选个风格点“开始剪辑”，剪完可以在这里继续改，比如“缩短到 10 秒”“换一首”“改成竖屏”。');
  }
  async function send(preset) {
    if (state.busy || !state.video) return;
    const text = (preset || $('#msg').value).trim();
    if (!text) return;
    addMsg('user', text);
    $('#msg').value = ''; $('#msg').style.height = '';
    stopPreview(); setBusy(true);
    const typing = addMsg('agent', '');
    typing.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
    const slow = setTimeout(() => working('CHAT'), 1500);  // quick answers skip the overlay
    try {
      const r = await api('/api/v1/chat', {
        message: text, session_id: state.sessionId || undefined, video_id: state.video.id,
        mode: state.mode, edit_style: editStyle(), selected_segment: state.selected || undefined
      });
      typing.remove();
      renderSuggest(r.suggestions);
      if (r.output) selectSegment(null);
      state.sessionId = r.session_id || state.sessionId;
      applyStyle(r.edit_style);
      addMsg('agent', r.reply || '好了。', r.output ? '<br><button class="play" type="button" data-play="1">▶ 看新版本</button>' : '');
      if (r.output) {
        working(null); adoptOutput(r.output);
        if (r.output.duration_note) toast(r.output.duration_note);
      }
      if (r.output || r.events || state.mode === 'ai') loadMoments();
    } catch (e) {
      typing.remove();
      if (e.code === 'VIP_REQUIRED') { lostVip(); openVip(); addMsg('system', 'AI 剪辑师需要 VIP，已切回基础剪辑。'); }
      else addMsg('system', e.handled ? e.message + '（基础剪辑仍可免费使用）' : e.message);
    } finally {
      clearTimeout(slow); working(null); setBusy(false); if (state.mode === 'ai') loadMe();
    }
  }
  /* ---------- share / save to album ---------- */
  // Fetch the reel ahead of time: iOS only opens the share sheet if share()
  // runs right inside the tap, with no download in between.
  let shareFile = null, shareFor = '';
  function prepareShare(url) {
    shareFile = null; shareFor = url;
    const btn = $('#share');
    btn.classList.add('hidden');
    if (!navigator.canShare || !window.matchMedia('(pointer: coarse)').matches) return;
    fetch(url).then((r) => r.ok ? r.blob() : Promise.reject()).then((blob) => {
      if (shareFor !== url) return;
      const file = new File([blob], 'MoTrack集锦.mp4', { type: 'video/mp4' });
      if (navigator.canShare({ files: [file] })) { shareFile = file; btn.classList.remove('hidden'); }
    }).catch(() => {});
  }
  async function share() {
    if (!shareFile) return;
    try {
      await navigator.share({ files: [shareFile], title: 'MoTrack 集锦' });
    } catch (e) {
      if (e && e.name !== 'AbortError') toast('分享失败，可以点“下载”保存', true);
    }
  }

  /* ---------- phone access (desktop only) ---------- */
  function phoneView(info) {
    $('#phoneErr').classList.add('hidden');
    $('#phoneWait').classList.add('hidden');
    const on = !!(info && info.url);
    $('#phoneOff').classList.toggle('hidden', on);
    $('#phoneOn').classList.toggle('hidden', !on);
    if (on) {
      $('#phoneQr').src = info.qr; $('#phoneUrl').textContent = info.url; $('#phoneUrl').href = info.url;
      $('#phoneCode').textContent = info.code;
    }
    if (info && info.error && !on) { $('#phoneErr').textContent = info.error; $('#phoneErr').classList.remove('hidden'); }
    if (info && !info.available) { $('#phoneStart').disabled = true; }
  }
  async function openPhone() {
    $('#phoneDialog').classList.remove('hidden');
    try { phoneView(await api('/api/v1/remote')); } catch (e) { phoneView({ error: e.message }); }
  }
  async function startPhone() {
    $('#phoneOff').classList.add('hidden'); $('#phoneWait').classList.remove('hidden');
    try {
      let info = await api('/api/v1/remote/start', {});
      for (let i = 0; i < 40 && info.running && !info.url; i++) {  // cloudflared needs a few seconds for its URL
        await new Promise((r) => setTimeout(r, 700));
        info = await api('/api/v1/remote');
      }
      if (!info.url && !info.error) info.error = '开启超时，请再试一次';
      phoneView(info);
    } catch (e) { phoneView({ error: e.message }); }
  }
  async function resetPhone() {
    if (!confirm('重置后旧二维码立即失效，已登录的手机都需要重新扫码。确定吗？')) return;
    try { phoneView(await api('/api/v1/remote/reset', {})); toast('已换成新的二维码'); } catch (e) { toast(e.message, true); }
  }
  async function stopPhone() {
    try { phoneView(await api('/api/v1/remote/stop', {})); toast('已关闭手机访问'); } catch (e) { toast(e.message, true); }
  }
  async function checkPhone() {
    // The panel exists only for this computer's own browser.
    try { await api('/api/v1/remote'); $('#phoneBtn').classList.remove('hidden'); } catch (e) { /* phone or old backend */ }
  }

  /* ---------- install to home screen ---------- */
  // Only Android Chrome/Edge fire beforeinstallprompt.  Everywhere else
  // (iPhone, WeChat, phone-maker browsers) the button opens step-by-step
  // instructions for the browser actually in use.
  let installEvent = null;
  const standalone = () => window.matchMedia('(display-mode: standalone)').matches || navigator.standalone === true;
  const isPhone = () => window.matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
  function browserInfo() {
    const ua = navigator.userAgent;
    const ios = /iP(hone|ad|od)/.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
    const name = /MicroMessenger/i.test(ua) ? 'wechat' : /\bQQ\//.test(ua) ? 'qq' : /AlipayClient/i.test(ua) ? 'alipay'
      : /Weibo/i.test(ua) ? 'weibo' : /DingTalk/i.test(ua) ? 'dingtalk' : /Lark|Feishu/i.test(ua) ? 'feishu'
      : /MiuiBrowser|XiaoMi\//i.test(ua) ? 'mi' : /HuaweiBrowser|HONORBrowser/i.test(ua) ? 'huawei'
      : /Quark/i.test(ua) ? 'quark' : /UCBrowser/i.test(ua) ? 'uc' : /MQQBrowser/i.test(ua) ? 'qqbrowser'
      : /HeyTapBrowser|OppoBrowser/i.test(ua) ? 'oppo' : /VivoBrowser/i.test(ua) ? 'vivo' : /SamsungBrowser/i.test(ua) ? 'samsung'
      : /EdgiOS|EdgA/i.test(ua) ? 'edge' : /CriOS|Chrome\//i.test(ua) ? 'chrome' : /FxiOS|Firefox/i.test(ua) ? 'firefox'
      : ios ? 'safari' : 'other';
    return { ios, name, inApp: ['wechat', 'qq', 'alipay', 'weibo', 'dingtalk', 'feishu'].includes(name) };
  }
  const APP_NAME = { wechat: '微信', qq: 'QQ', alipay: '支付宝', weibo: '微博', dingtalk: '钉钉', feishu: '飞书' };
  function installSteps() {
    const b = browserInfo();
    if (b.inApp) {
      return { title: APP_NAME[b.name] + '里不能安装，先换到浏览器', copy: true, steps: [
        '点右上角的 <b>···</b>',
        b.ios ? '选 <b>“在 Safari 中打开”</b>' : '选 <b>“在浏览器打开”</b>',
        '打开后，再点页面右上角的 <b>📲 安装</b>，按提示操作'] };
    }
    if (b.ios) {
      if (['safari', 'chrome', 'edge', 'firefox'].includes(b.name)) {
        return { title: '添加到主屏幕', steps: [
          '点地址栏旁边的 <b>分享按钮</b>（方框加向上箭头 ⬆️；新版 iOS 可能要先点 <b>···</b>）',
          '往下滑，选 <b>“添加到主屏幕”</b>',
          '点右上角 <b>“添加”</b>，主屏幕上就有 MoTrack 图标了'],
          note: b.name === 'safari' ? '' : '没有“添加到主屏幕”的话，用 Safari 打开这个网址再试。' };
      }
      return { title: '用 Safari 打开才能添加', copy: true, steps: ['复制网址', '打开 <b>Safari</b>，粘贴网址打开', '点 <b>分享 ⬆️</b> → <b>“添加到主屏幕”</b>'] };
    }
    const menu = { mi: '<b>☰</b>（底部菜单）→ <b>“添加到桌面”</b>', huawei: '<b>⋮</b> 或底部 <b>☰</b> → <b>“添加至桌面”</b>',
      quark: '底部 <b>☰</b> → <b>“添加到桌面”</b>', uc: '底部 <b>☰</b> → <b>“添加到桌面”</b>', qqbrowser: '底部 <b>☰</b> → <b>“添加到桌面”</b>',
      oppo: '<b>⋮</b> → <b>“添加到桌面”</b>', vivo: '<b>☰</b> → <b>“添加到桌面”</b>', samsung: '<b>☰</b> → <b>“添加页面到”</b> → <b>“主屏幕”</b>',
      chrome: '右上角 <b>⋮</b> → <b>“添加到主屏幕”</b> 或 <b>“安装应用”</b>', edge: '底部 <b>···</b> → <b>“添加到手机”</b>',
      firefox: '<b>⋮</b> → <b>“安装”</b> 或 <b>“添加到主屏幕”</b>' }[b.name] || '浏览器菜单（<b>⋮</b> 或 <b>☰</b>）→ <b>“添加到桌面 / 主屏幕”</b>';
    return { title: '添加到桌面', steps: ['点 ' + menu, '确认添加，桌面上就有 MoTrack 图标了'],
      note: '小米、华为、OPPO、vivo 手机如果没反应：到系统设置 → 应用 → 这个浏览器 → 权限，打开 <b>“桌面快捷方式”</b>。还不行就用 Chrome 打开。' };
  }
  function openInstall() {
    const s = installSteps();
    $('#installTitle').textContent = s.title;
    $('#installSteps').innerHTML = s.steps.map((t) => '<li>' + t + '</li>').join('');
    $('#installNote').innerHTML = s.note || '';
    $('#installNote').classList.toggle('hidden', !s.note);
    $('#installCopy').classList.toggle('hidden', !s.copy);
    $('#installDialog').classList.remove('hidden');
  }
  async function install() {
    if (installEvent) {  // Android Chrome / Edge: the real install prompt
      const event = installEvent; installEvent = null;
      try {
        await event.prompt();
        if ((await event.userChoice).outcome === 'accepted') return;
      } catch (e) { /* prompt refused or dismissed: show the manual steps */ }
    }
    openInstall();
  }
  async function copyLink() {
    const url = location.origin + '/app/index.html';
    try { await navigator.clipboard.writeText(url); toast('网址已复制'); }
    catch (e) { prompt('长按复制这个网址：', url); }
  }
  window.addEventListener('beforeinstallprompt', (e) => { e.preventDefault(); installEvent = e; });
  window.addEventListener('appinstalled', () => { $('#installBtn').classList.add('hidden'); $('#installTip').classList.add('hidden'); toast('已添加到主屏幕 🎉'); });
  function installUI() {
    if (standalone() || !isPhone()) return;
    $('#installBtn').classList.remove('hidden');
    let seen = false;
    try { seen = localStorage.getItem('motrack_install_tip') === '1'; } catch (e) { /* private mode */ }
    if (!seen) {
      const b = browserInfo();
      $('#installTipText').innerHTML = b.inApp ? '在' + APP_NAME[b.name] + '里打开的？点右上角 <b>···</b> 换到浏览器，就能把 MoTrack 放到桌面。'
        : '把 MoTrack 放到桌面，像 App 一样打开。';
      $('#installTip').classList.remove('hidden');
    }
  }
  if ('serviceWorker' in navigator && window.isSecureContext) {
    navigator.serviceWorker.register('/app/sw.js', { scope: '/app/' }).catch(() => {});
  }

  /* ---------- bindings ---------- */
  $('#logo').src = $('#favicon').href;
  $('#videos').addEventListener('click', (e) => {
    const del = e.target.closest('[data-del]');
    if (del) { e.stopPropagation(); deleteVideo(del.dataset.del); return; }
    if (e.target.closest('#uploadCard')) { if (!state.busy) $('#videoFile').click(); return; }
    const b = e.target.closest('.vcard[data-id]'); if (b) selectVideo(b.dataset.id);
  });
  $('#videoFile').addEventListener('change', (e) => uploadVideo(e.target.files[0]));
  $('#share').addEventListener('click', share);
  $('#phoneBtn').addEventListener('click', openPhone);
  $('#phoneStart').addEventListener('click', startPhone);
  $('#phoneStop').addEventListener('click', stopPhone);
  $('#phoneReset').addEventListener('click', resetPhone);
  $('#phoneClose').addEventListener('click', () => $('#phoneDialog').classList.add('hidden'));
  $('#phoneDialog').addEventListener('click', (e) => { if (e.target === e.currentTarget) $('#phoneDialog').classList.add('hidden'); });
  $('#installBtn').addEventListener('click', install);
  $('#installTipGo').addEventListener('click', install);
  $('#installTipClose').addEventListener('click', () => {
    $('#installTip').classList.add('hidden');
    try { localStorage.setItem('motrack_install_tip', '1'); } catch (e) { /* private mode */ }
  });
  $('#installCopy').addEventListener('click', copyLink);
  $('#installClose').addEventListener('click', () => $('#installDialog').classList.add('hidden'));
  $('#installDialog').addEventListener('click', (e) => { if (e.target === e.currentTarget) $('#installDialog').classList.add('hidden'); });
  $$('.mode').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  $('#presets').addEventListener('click', (e) => { const b = e.target.closest('.preset'); if (b) { stopPreview(); applyPreset(b.dataset.id); } });
  $('#durations').addEventListener('click', (e) => {
    const b = e.target.closest('.chip'); if (!b) return;
    state.duration = Number(b.dataset.d); renderDurations();
  });
  ['aspect', 'focus'].forEach((id) => $('#' + id).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (b) setSeg(id, b.dataset.value);
  }));
  ['pace', 'transition', 'look'].forEach((id) => $('#' + id).addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (b) { setSeg(id, b.dataset.value); matchPreset(); }
  }));
  ['#zoom', '#slowmo'].forEach((s) => $(s).addEventListener('change', matchPreset));
  $('#music').addEventListener('click', (e) => {
    const b = e.target.closest('.chip'); if (!b) return;
    stopPreview(); state.music = b.dataset.id; renderMusic(); matchPreset();
  });
  $('#musicPreview').addEventListener('click', previewMusic);
  $('#musicAudio').addEventListener('ended', stopPreview);
  $('#musicFile').addEventListener('change', (e) => uploadMusic(e.target.files[0]));
  $('#title').addEventListener('input', () => { state.titleOff = false; });
  $$('#viewToggle button').forEach((b) => b.addEventListener('click', () => showView(b.dataset.view)));
  $('#moments').addEventListener('click', (e) => {
    const b = e.target.closest('.moment'); if (!b) return;
    seekSource(b.dataset.t);
  });
  const reviewBtn = $('#reviewBtn'); if (reviewBtn) reviewBtn.addEventListener('click', openReview);
  $('#reviewClose').addEventListener('click', () => $('#reviewDialog').classList.add('hidden'));
  $('#reviewDialog').addEventListener('click', (e) => {
    if (e.target.id === 'reviewDialog') { $('#reviewDialog').classList.add('hidden'); return; }
    if (e.target.closest('#recapGo')) { makeRecap(); return; }
    const bg = e.target.closest('#bestGo');
    if (bg) { $('#reviewDialog').classList.add('hidden'); showPane('chat'); send('把本场最精彩的' + bg.dataset.n + '个进球剪成集锦'); return; }
    const pl = e.target.closest('[data-player]');
    if (pl) { $('#reviewDialog').classList.add('hidden'); send('只看' + pl.dataset.player + '号的集锦'); return; }
    const b = e.target.closest('[data-t]'); if (!b) return;
    $('#reviewDialog').classList.add('hidden'); seekSource(b.dataset.t);
  });
  $('#filmTrack').addEventListener('click', (e) => {
    const b = e.target.closest('.seg-item'); if (!b) return;
    if (b.dataset.n) selectSegment(Number(b.dataset.n) === state.selected ? null : Number(b.dataset.n));
    showView('highlight');
    const v = $('#highlight'); v.currentTime = Number(b.dataset.out) + 0.05; v.play().catch(() => {});
  });
  $('#log').addEventListener('click', (e) => {
    if (!e.target.closest('[data-play]')) return;
    showView('highlight'); const v = $('#highlight'); v.currentTime = 0; v.play().catch(() => {});
    $('.player').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
  $$('.tabs button').forEach((b) => b.addEventListener('click', () => showPane(b.dataset.pane)));
  $('#askAi').addEventListener('click', () => { showPane('chat'); $('#msg').focus(); });
  $('#go').addEventListener('click', generate);
  $('#send').addEventListener('click', () => send());
  $('#suggest').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) send(b.textContent); });
  $('#newChat').addEventListener('click', () => { if (!state.busy) resetChat(); });
  $('#msg').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing && e.keyCode !== 229) { e.preventDefault(); send(); }
  });
  $('#msg').addEventListener('input', () => { const t = $('#msg'); t.style.height = 'auto'; t.style.height = Math.min(110, t.scrollHeight) + 'px'; });
  $('#vipPill').addEventListener('click', openVip);
  $('#helpBtn').addEventListener('click', openGuide);
  $('#guideSkip').addEventListener('click', closeGuide);
  $('#guideNext').addEventListener('click', () => { if (guideAt >= guideList.length - 1) closeGuide(); else { guideAt++; showGuideStep(); } });
  window.addEventListener('resize', placeGuide);
  $('#selClear').addEventListener('click', () => selectSegment(null));
  $('#accessClose').addEventListener('click', () => $('#accessDialog').classList.add('hidden'));
  $('#accessDialog').addEventListener('click', (e) => { if (e.target === e.currentTarget) $('#accessDialog').classList.add('hidden'); });
  $('#accessSend').addEventListener('click', applyAccess);
  $('#accessName').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) applyAccess(); });
  $('#quota').addEventListener('click', (e) => { if (e.target.closest('#quotaApply')) openAccess(); });
  $('#vipClose').addEventListener('click', closeVip);
  $('#vipDialog').addEventListener('click', (e) => { if (e.target === e.currentTarget) closeVip(); });
  $('#vipActivate').addEventListener('click', activate);
  $('#vipCode').addEventListener('keydown', (e) => { if (e.key === 'Enter') activate(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { closeVip(); $('#phoneDialog').classList.add('hidden'); $('#installDialog').classList.add('hidden'); $('#reviewDialog').classList.add('hidden'); $('#accessDialog').classList.add('hidden'); if (!$('#guide').classList.contains('hidden')) closeGuide(); } });

  (async function init() {
    applyPreset('hype'); renderMode(); resetChat(); renderSteps();
    await Promise.all([checkBackend(), checkVip(), loadMusic(), loadMe()]);
    applyPreset('hype'); restorePrefs(); renderDurations(); renderMode(); resetChat(); installUI(); checkPhone();
    await loadVideos();
    maybeGuide();
  })();
})();
