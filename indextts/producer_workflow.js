(encoded) => {
  const incoming = JSON.parse(encoded || '{}');
  if (!incoming.workspaceId) return;
  let workflow = document._producerWorkflow;
  if (!workflow) {
    workflow = document._producerWorkflow = {clientId:crypto.randomUUID(), sequence:0, context:null, timer:null};
    const field = id => document.querySelector(`#${id} textarea, #${id} input:not([type=range])`);
    const put = (id,value) => {
      const input = field(id); if (!input) return;
      Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set.call(input,value);
      input.dispatchEvent(new Event('input',{bubbles:true}));
    };
    workflow.collect = () => {
      const d = workflow.context;
      const root = document.querySelector('#material-waveform');
      let editor = d.editor;
      if (root && JSON.parse(root.dataset.config).workspaceId === d.workspaceId) {
        editor = root.dataset.editorState ? JSON.parse(root.dataset.editorState) : d.editor;
      }
      const number = (id,fallback) => Number(field(id)?.value ?? fallback);
      const radio = (id,fallback) => document.querySelector(`#${id} input:checked`)?.value || fallback;
      // Gradio custom dropdown exposes its selected label in a listbox input.
      const dropdown = (id,fallback) => document.querySelector(`#${id} input[role=listbox]`)?.value || fallback;
      const gender = {'未知':'unknown','男声':'male','女声':'female'}[dropdown('producer-gender','')] || d.gender;
      const profile = {'FP32':'compatible-fp32','BF16':'fixed-voice-bf16'}[radio('producer-profile','')] || radio('producer-profile',d.profile);
      const device = {'自动':'auto','CPU':'cpu'}[dropdown('producer-device','')] || dropdown('producer-device',d.device);
      const vector = Array.from({length:8},(_,i)=>number(`producer-emotion-${i}`,Array.isArray(d.emotion) ? d.emotion[i] : 0));
      return JSON.stringify({workspaceId:d.workspaceId, revision:d.revision, clientId:workflow.clientId,
        sequence:++workflow.sequence, editor, name:field('producer-name')?.value ?? d.name,
        text:field('producer-text')?.value ?? d.text, emotion:vector.some(v=>v!==0) ? vector : 'base', gender, profile, device,
        filters:{silence:number('material-silence',d.filters.silence), minimum:number('material-minimum',d.filters.minimum),
          volume:number('material-volume',d.filters.volume), enabled:!!field('material-volume-enabled')?.checked}});
    };
    workflow.signature = d => {
      const disabled=new Set(d.editor?.disabled || []), rows=d.editor?.rows?.filter(r=>r[3] && !disabled.has(r[0])) || [];
      const primary=rows.find(r=>r[0]===d.editor?.primary)?.[0] ||
        rows.reduce((best,r)=>!best || r[2]-r[1]>best[2]-best[1] ? r : best,null)?.[0];
      return JSON.stringify([d.workspaceId,d.name,d.gender,d.editor?.sourceId,rows,primary]);
    };
    workflow.schedule = () => {
      clearTimeout(workflow.timer);
      document.querySelector('#producer-download-file')?.classList.add('draft-dirty');
      // Local recovery covers a refresh inside the 500ms server-save window.
      queueMicrotask(() => {
        const value=workflow.collect(), draft=JSON.parse(value), f=draft.filters;
        if(Number.isFinite(f.silence+f.minimum+f.volume) && f.silence>=.1 && f.silence<=3
            && f.minimum>=.1 && f.minimum<=15 && f.volume>=-80 && f.volume<=0 && draft.name.length<=128
            && (draft.emotion==='base' || draft.emotion.every(v=>Number.isFinite(v) && v>=0 && v<=1.2)))
          localStorage.setItem(`producer-pending:${draft.workspaceId}`,value);
      });
      workflow.timer=setTimeout(()=>put('producer-save-payload',workflow.collect()),500);
    };
    document.addEventListener('producer-editor-change',workflow.schedule);
    document.addEventListener('input',event=>{
      if(event.target.closest('#producer-name,#producer-gender,#producer-profile,#producer-device,#producer-text,#producer-emotion,#material-silence,#material-minimum,#material-volume,#material-volume-enabled')) workflow.schedule();
    });
    document.addEventListener('change',event=>{
      if(event.target.closest('#producer-gender,#producer-profile,#producer-device')) workflow.schedule();
    });
    document.addEventListener('click',event=>{
      if(event.target.closest('#producer-gender,#producer-device')) workflow.schedule();
      if(event.target.closest('#producer-export')) workflow.exportSignature=workflow.signature(JSON.parse(workflow.collect()));
    });
  }
  clearTimeout(workflow.timer);
  workflow.context=incoming;
  localStorage.setItem('producer-active-workspace',incoming.workspaceId);
  const pending=JSON.parse(localStorage.getItem(`producer-pending:${incoming.workspaceId}`) || 'null');
  if(pending && incoming.clients?.[pending.clientId]>=pending.sequence)
    localStorage.removeItem(`producer-pending:${incoming.workspaceId}`);
  return [];
}
