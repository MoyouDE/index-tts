() => {
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const root = document.getElementById('material-waveform');
    if (!root || root._editor) return;
    root._editor = true;
    const config = JSON.parse(root.dataset.config);
    const svg = root.querySelector('svg'), audio = root.querySelector('audio');
    const info = root.querySelector('.info'), clock = root.querySelector('.clock');
    const pan = root.querySelector('.pan'), ns = 'http://www.w3.org/2000/svg';
    let rows = structuredClone(config.rows), primary = config.primary;
    let active = rows.find(r => r[3] && r[0] === primary)?.[0] || rows.find(r => r[3])?.[0];
    let zoom = 1, offset = 0, drag = null, stop = null, frame = null, history = [];
    const duration = config.duration;
    const clamp = (t, a = 0, b = duration) => Math.min(b, Math.max(a, t));
    const fmt = t => `${Math.floor(t / 60)}:${(t % 60).toFixed(2).padStart(5, '0')}`;
    const x = t => (t - offset) / (duration / zoom) * 1000;
    const at = event => {
      const box = svg.getBoundingClientRect();
      return clamp(Math.round((offset + (event.clientX - box.left) / box.width * duration / zoom)*1000)/1000);
    };
    const chosen = () => rows.filter(r => r[3]).sort((a,b) => a[1]-b[1]);
    const current = () => rows.find(r => r[3] && r[0] === active);
    const effectivePrimary = () => chosen().find(r => r[0] === primary)?.[0] ||
      chosen().reduce((best,r) => !best || r[2]-r[1] > best[2]-best[1] ? r : best, null)?.[0];
    const button = name => root.querySelector(`[data-action="${name}"]`);
    const scale = 58 / Math.max(.05,...config.peaks.low.map(Math.abs),...config.peaks.high.map(Math.abs));
    function publish() {
      const field = document.querySelector('#material-waveform-draft textarea');
      if (!field) { info.textContent = '页面尚未准备好，请重新选择素材'; return; }
      const value = JSON.stringify({sourceId:config.sourceId, rows, primary});
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(field,value);
      field.dispatchEvent(new Event('input',{bubbles:true}));
    }
    function el(tag, attrs, parent = svg) {
      const node = document.createElementNS(ns,tag);
      for (const [key,value] of Object.entries(attrs)) node.setAttribute(key,String(value));
      parent.appendChild(node); return node;
    }
    function draw() {
      svg.replaceChildren();
      const {low, high} = config.peaks, n = low.length;
      const begin = Math.max(0,Math.floor(offset/duration*n));
      const end = Math.min(n,Math.ceil((offset+duration/zoom)/duration*n));
      let d = '';
      const stride = Math.max(1,Math.floor((end-begin)/1000));
      for (let i=begin;i<end;i+=stride) {
        const px=x(i/n*duration), lo=Math.min(...low.slice(i,Math.min(end,i+stride))), hi=Math.max(...high.slice(i,Math.min(end,i+stride)));
        d += `M${px.toFixed(2)},${(102-hi*scale).toFixed(2)}V${(102-lo*scale).toFixed(2)}`;
      }
      el('path',{d,stroke:'#8581b4','stroke-width':1,fill:'none','pointer-events':'none'});
      for (let i=0;i<=5;i++) {
        const t=offset+i*duration/zoom/5, px=i*200;
        el('line',{x1:px,x2:px,y1:28,y2:174,stroke:'#ebecf5','pointer-events':'none'});
        const tick=el('text',{x:clamp(px,2,950),y:186,fill:'#77788e','font-size':11,'pointer-events':'none'}); tick.textContent=fmt(t);
      }
      chosen().forEach((r,i) => {
        if (x(r[2])<0 || x(r[1])>1000) return;
        const start=clamp(x(r[1]),0,1000), end=clamp(x(r[2]),0,1000), focused=r[0]===active;
        const region=el('g',{'data-id':r[0],role:'button','aria-label':`片段 ${i+1}，${r[1].toFixed(3)} 至 ${r[2].toFixed(3)} 秒`,tabindex:0});
        el('rect',{x:start,y:27,width:Math.max(1,end-start),height:145,rx:3,fill:focused?'#695aff':'#9691d0','fill-opacity':focused?.22:.12,stroke:focused?'#695aff':'#9691d0','data-id':r[0]},region);
        if (end-start>18) { const label=el('text',{x:start+4,y:42,fill:'#554bb2','font-size':12,'pointer-events':'none'},region); label.textContent=`${i+1}${r[0]===effectivePrimary()?' ★':''}`; }
        if (focused) {
          for (const [side,t] of [['start',r[1]],['end',r[2]]]) {
            if (x(t)<0 || x(t)>1000) continue;
            el('rect',{x:clamp(x(t)-5,0,990),y:50,width:10,height:105,rx:3,fill:'#695aff','data-id':r[0],'data-edge':side,style:'cursor:ew-resize'},region);
          }
        }
      });
      el('line',{x1:x(audio.currentTime),x2:x(audio.currentTime),y1:22,y2:175,stroke:'#ee8657','stroke-width':2,'pointer-events':'none','data-playhead':''});
      const r=current(), list=chosen();
      info.textContent=`已选 ${list.length} 段 · ${list.reduce((s,r)=>s+r[2]-r[1],0).toFixed(2)} 秒${r ? `　当前 ${fmt(r[1])}–${fmt(r[2])}（${(r[2]-r[1]).toFixed(2)} 秒）${r[0]===effectivePrimary()?' · 主参考':''}`:''}`;
      button('preview').disabled=button('delete').disabled=button('primary').disabled=!r;
      button('undo').disabled=!history.length;
      button('out').disabled=zoom===1; button('in').disabled=zoom>=64;
      pan.hidden=zoom===1; pan.max=Math.max(0,duration-duration/zoom); pan.value=offset;
      clock.textContent=`${fmt(audio.currentTime)} / ${fmt(duration)} · ${zoom}×`;
    }
    function commit(before, oldPrimary) {
      const list=chosen();
      let error = list.some(r=>r[1]<0 || r[2]>duration+1e-6 || r[2]<=r[1]) ? '片段边界必须在音轨内且起点小于终点' :
        list.some(r=>r[2]-r[1]>15+1e-6) ? '每段最多 15 秒，请缩短范围或分段拖选' :
        list.some((r,i)=>i && list[i-1][2]>r[1]+1e-6) ? '片段重叠，请在空白处选择或调整边界' : null;
      if (error) { rows=before; primary=oldPrimary; draw(); info.textContent=error; return; }
      history.push({rows:before,primary:oldPrimary}); if(history.length>30) history.shift();
      rows.sort((a,b)=>a[1]-b[1]); publish(); draw();
    }
    function playbackTick() {
      if (!root.isConnected) { audio.pause(); return; }
      if (stop!==null && audio.currentTime>=stop) { const end=stop; audio.pause(); audio.currentTime=end; }
      const head=svg.querySelector('[data-playhead]'); if(head) {head.setAttribute('x1',x(audio.currentTime));head.setAttribute('x2',x(audio.currentTime));}
      clock.textContent=`${fmt(audio.currentTime)} / ${fmt(duration)} · ${zoom}×`;
      if(!audio.paused) frame=requestAnimationFrame(playbackTick);
    }
    function play(start=null,end=null) {
      audio.pause(); if(start!==null) audio.currentTime=start; stop=end;
      audio.play().catch(e=>{info.textContent=`无法播放：${e.message}`;stop=null;});
    }
    audio.addEventListener('play',()=>{button('play').textContent='❚❚ 暂停'; cancelAnimationFrame(frame);playbackTick();});
    audio.addEventListener('pause',()=>{button('play').textContent='▶ 播放';stop=null;cancelAnimationFrame(frame);});
    audio.addEventListener('timeupdate',()=>{if(audio.paused)playbackTick();else if(stop!==null && audio.currentTime>=stop){const end=stop;audio.pause();audio.currentTime=end;}});
    audio.addEventListener('error',()=>{info.textContent='音轨播放失败，请检查素材文件是否可读取';});
    svg.addEventListener('pointerdown',event=>{
      if(event.button!==0) return;
      audio.pause(); svg.focus(); svg.setPointerCapture(event.pointerId);
      const target=event.target.closest('[data-id]'), row=target && rows.find(r=>r[0]===target.dataset.id);
      active=row?.[0];
      drag={time:at(event),pixel:event.clientX,before:structuredClone(rows),primary,mode:row ? (event.target.dataset.edge||'move'):'new',row:row ? [...row]:null,moved:false};
      draw();
    });
    svg.addEventListener('pointermove',event=>{
      if(!drag) return;
      if(Math.abs(event.clientX-drag.pixel)<3 && !drag.moved) return;
      drag.moved=true; const t=at(event);
      if(drag.mode==='new') {
        if(!drag.id) {drag.id='s'+crypto.randomUUID().replaceAll('-','').slice(0,10); rows.push([drag.id,0,0,true]);active=drag.id;}
        const r=rows.find(r=>r[0]===drag.id); r[1]=Math.min(t,drag.time);r[2]=Math.max(t,drag.time);
      } else {
        const r=current(); if(!r) return;
        if(drag.mode==='start') r[1]=t;
        else if(drag.mode==='end') r[2]=t;
        else {const delta=clamp(t-drag.time,-drag.row[1],duration-drag.row[2]);r[1]=drag.row[1]+delta;r[2]=drag.row[2]+delta;}
      }
      draw();
    });
    svg.addEventListener('pointerup',event=>{
      if(!drag) return;
      const saved=drag; drag=null;
      if(saved.moved) commit(saved.before,saved.primary);
      else {audio.currentTime=saved.time;draw();}
      if(svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
    });
    svg.addEventListener('pointercancel',()=>{if(drag){rows=drag.before;primary=drag.primary;drag=null;draw();}});
    svg.addEventListener('dblclick',event=>{const region=event.target.closest('[data-id]');if(region){active=region.dataset.id;const r=current();if(r)play(r[1],r[2]);draw();}});
    root.addEventListener('click',event=>{
      const action=event.target.closest('[data-action]')?.dataset.action; if(!action) return;
      const r=current(), before=structuredClone(rows), oldPrimary=primary;
      if(action==='play') {if(audio.paused)play();else audio.pause();}
      if(action==='preview'&&r) play(r[1],r[2]);
      if(action==='delete'&&r) {r[3]=false;active=null;commit(before,oldPrimary);}
      if(action==='primary'&&r) {primary=r[0];commit(before,oldPrimary);}
      if(action==='undo'&&history.length) {const previous=history.pop();rows=previous.rows;primary=previous.primary;active=chosen()[0]?.[0];publish();draw();}
      if(action==='in'||action==='out') {
        const center=r ? (r[1]+r[2])/2 : offset+duration/zoom/2;
        zoom=clamp(zoom*(action==='in'?2:.5),1,64);offset=clamp(center-duration/zoom/2,0,duration-duration/zoom);draw();
      }
    });
    svg.addEventListener('keydown',event=>{
      const id=event.target.closest('[data-id]')?.dataset.id;
      if(id) active=id;
      if(event.code==='Space'){event.preventDefault();button('play').click();}
      if(event.code==='Delete'||event.code==='Backspace'){event.preventDefault();button('delete').click();}
      if(event.code==='Enter'&&id){event.preventDefault();draw();button('preview').click();}
    });
    pan.addEventListener('input',()=>{offset=Number(pan.value);draw();});
    draw(); publish();
  }));
  return [];
}
