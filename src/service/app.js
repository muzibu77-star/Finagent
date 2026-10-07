'use strict';
const $ = id => document.getElementById(id);
let catalogue, taskId = localStorage.getItem('finagent.task'), cursor = 0, current, polling = false;
const names = {queued:'已排队',running:'正在研究',generation_started:'正在分析问题与证据',generation_finished:'分析完成',tool_observation:'工具执行完成',tool_rejected:'调用格式不符，已记录',clarification:'需要补充条件',report_committed:'报告已保存',cancel_requested:'已请求取消',cancelled:'已取消',failed:'执行失败',recovered:'已恢复任务',succeeded:'等待补充条件',worker_health:'工作进程健康检查'};
const tools = {search_documents:'检索资料',get_evidence:'读取原文',calculate:'执行计算'};
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) {const error=new Error(data.error || `请求失败 ${response.status}`);error.status=response.status;throw error;}
  return data;
}
function node(tag,text,className) { const el=document.createElement(tag);el.textContent=text;if(className)el.className=className;return el; }
function showError(error) {$('error').textContent=error.message;}
function taskPath(suffix='') {return '/api/tasks/'+encodeURIComponent(taskId)+suffix;}
function renderReport(task) {
  const root=$('report');root.replaceChildren();$('export').disabled=!task.report;
  if (!task.report) {root.append(node('p',task.execution==='failed'?'本轮未完成，未生成报告。':'本轮尚无报告。','empty'));return;}
  if(task.business==='insufficient_evidence') {root.append(node('p','当前公司、期间及截止条件下，资料不足。','empty'));return;}
  root.append(node('p','显示计算原值，未自动换算单位或百分数；请核对引用资料中的指标和期间。','hint'));
  for(const calculation of task.report.calculations) {
    const box=node('div','', 'result');box.append(node('div',String(calculation.value),'value'));
    const links=node('div','', 'citations');
    const ids=[...new Set(Object.values(calculation.facts).map(f=>f.evidence_id))];
    for(const id of ids) {const button=node('button',id,'secondary');button.onclick=()=>showSource(id).catch(showError);links.append(button);}
    box.append(links);root.append(box);
  }
}
async function showSource(id) {
  const source=await api(taskPath('/source?id='+encodeURIComponent(id)));
  $('sourceMeta').replaceChildren(node('span',source.source_report_id+(source.pdf_page?' · PDF 第 '+source.pdf_page+' 页':'')));
  if(source.url && source.url.startsWith('https://')) {const link=node('a',' 打开官方原文');link.href=source.url;link.target='_blank';link.rel='noopener noreferrer';$('sourceMeta').append(link);}
  $('sourceText').textContent=JSON.stringify(source.evidence,null,2);$('source').showModal();
}
async function poll() {
  if(polling)return;polling=true;
  try {
    catalogue=await api('/api/catalogue');$('connection').textContent=catalogue.ready?'模型已就绪':catalogue.error?'工作进程已停止':'模型加载中';
    if(catalogue.ready && localStorage.getItem('finagent.pending'))await submitPending();
    if(taskId) {
      const task=await api(taskPath());const events=await api(taskPath('/events?after='+cursor));
      for(const event of events) {const text=event.kind==='tool_observation'?(tools[event.body.tool]||'工具执行完成'):(names[event.kind]||event.kind);$('events').append(node('li','第 '+event.revision+' 轮 · '+text));cursor=Math.max(cursor,event.seq);}
      const changed=!current || current.revision!==task.revision || current.execution!==task.execution || current.business!==task.business;
      current=task;$('status').textContent=task.business==='answered'?'已回答':task.business==='awaiting_input'?'等待补充条件':task.business==='insufficient_evidence'?'资料不足':names[task.execution]||task.execution;
      $('questionShown').textContent=task.payload.question;$('cancel').disabled=!['queued','running'].includes(task.execution);
      $('clarification').textContent=task.detail?.question||task.detail?.error||'';
      if(changed)renderReport(task);
    }
    $('submit').disabled=!catalogue.ready || Boolean(current && ['queued','running'].includes(current.execution));
  } catch(error) {showError(error);$('connection').textContent='连接中断，正在重试';}
  finally {polling=false;}
}
$('form').onsubmit=async event=>{event.preventDefault();$('error').textContent='';$('submit').disabled=true;
  try {
    if(localStorage.getItem('finagent.pending'))throw new Error('正在确认上次提交，请稍候。');
    const periods=$('period').value.split(',').map(x=>x.trim()).filter(Boolean);
    const payload={question:$('question').value,snapshot_id:catalogue.snapshot_id,company:$('company').value||null,period:periods.length>1?periods:periods[0]||null};
    if($('asOf').value)payload.as_of=$('asOf').value;
    const request={request_id:crypto.randomUUID(),payload};if(taskId)request.task_id=taskId;
    // Keep the same request ID if the response is lost; server input hashes bind retries.
    localStorage.setItem('finagent.pending',JSON.stringify(request));await submitPending();
  } catch(error) {showError(error);}finally{await poll();}
};
async function submitPending(){const pending=JSON.parse(localStorage.getItem('finagent.pending'));if(!pending)return;let result;try{result=await api('/api/tasks',pending);}catch(error){if(error.status>=400 && error.status<500)localStorage.removeItem('finagent.pending');throw error;}taskId=result.task_id;localStorage.setItem('finagent.task',taskId);localStorage.removeItem('finagent.pending');}
$('cancel').onclick=()=>api(taskPath('/cancel'),{}).then(poll).catch(showError);
$('new').onclick=()=>{taskId=null;current=null;cursor=0;localStorage.removeItem('finagent.task');localStorage.removeItem('finagent.pending');$('events').replaceChildren();$('questionShown').textContent='';$('clarification').textContent='';$('question').value='';$('status').textContent='等待提问';$('report').replaceChildren(node('p','完成后在此查看计算结果及引用原文。','empty'));$('export').disabled=true;poll();};
$('company').onchange=()=>{$('period').placeholder=(catalogue.companies[$('company').value]||[]).join(', ');};
$('closeSource').onclick=()=>$('source').close();
$('export').onclick=async()=>{try{const report=await api(taskPath('/report'));const url=URL.createObjectURL(new Blob([JSON.stringify(report,null,2)],{type:'application/json'}));const a=node('a','');a.href=url;a.download='finagent-'+report.task_id+'-v'+report.revision+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(error){showError(error);}};
(async()=>{try{catalogue=await api('/api/catalogue');$('notice').textContent=catalogue.notice;$('cutoffLabel').hidden=!catalogue.strict_pit;for(const company of Object.keys(catalogue.companies)){const option=node('option',company);option.value=company;$('company').append(option);}if(taskId){const task=await api(taskPath());$('company').value=task.payload.company||'';$('period').value=Array.isArray(task.payload.period)?task.payload.period.join(','):task.payload.period||'';$('asOf').value=task.payload.as_of||'';}if(localStorage.getItem('finagent.pending') && catalogue.ready)await submitPending();await poll();}catch(error){showError(error);}setInterval(poll,1000);})();
