(() => {
  const form = document.querySelector('#live-rec-form');
  const input = document.querySelector('#live-rec-query');
  const button = document.querySelector('#live-rec-submit');
  const status = document.querySelector('#live-rec-status');
  const results = document.querySelector('#live-rec-results');
  const planBox = document.querySelector('#live-rec-plan');
  const paperHeading = document.querySelector('#live-rec-paper-heading');
  const authorSection = document.querySelector('#live-rec-authors');
  const authorResults = document.querySelector('#live-rec-author-results');
  const authorJump = document.querySelector('#live-rec-author-jump');
  const names = {dtm:'D/T/M',graph:'인용 그래프',text:'제목·초록',e5:'의미 검색',canonical:'정규화 의미 검색'};
  const facets = {node_classification:'노드 분류',graph_learning:'그래프 학습',fpga:'FPGA',neural_network:'신경망',differential_privacy:'차등 프라이버시',collaborative_learning:'공동 학습',nonconvex:'비볼록',stochastic_gradient:'확률적 경사',implicit_feedback:'암묵적 피드백'};
  const conditions = {power:'전력 수치',throughput:'처리량',node_features:'노드 특성'};
  const exclusions = {survey:'서베이 제외',attack:'공격 연구 제외',convex_only:'볼록 전용 연구 제외'};
  function el(tag, text, cls) {
    const node = document.createElement(tag); if (text != null) node.textContent = text;
    if (cls) node.className = cls; return node;
  }
  function externalLink(label, url, cls) {
    const link = el('a', label, cls);
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    return link;
  }
  function paperSearch(title) {
    return 'https://arxiv.org/search/?searchtype=title&query=' + encodeURIComponent(title);
  }
  function renderAuthors(authors) {
    authorResults.replaceChildren();
    if (!authors.length) {
      authorResults.append(el('p', '검색 후보에서 저자 정보를 확인하지 못했습니다.', 'live-rec-muted'));
    }
    for (const author of authors) {
      const card = el('article', null, 'rec-author live-rec-author');
      card.append(el('h4', author.author_name), el('p', `상위 후보의 근거 논문 ${author.related_paper_count}편`));
      const evidence = el('ul');
      for (const paper of author.evidence_papers) {
        const row = el('li');
        row.append(paper.title ? externalLink(paper.title, paperSearch(paper.title)) : document.createTextNode(`논문 ${paper.paper_id}`));
        evidence.append(row);
      }
      card.append(evidence);
      if (/^https:\/\/openalex\.org\/A\d+$/.test(author.author_id)) {
        card.append(externalLink('OpenAlex 프로필 ↗', author.author_id, 'live-rec-link'));
      }
      authorResults.append(card);
    }
  }
  function render(data) {
    planBox.replaceChildren(); planBox.hidden = false;
    planBox.append(el('strong','해석한 검색어'),el('p',data.plan.positive_query || '인식된 검색어가 없습니다.'));
    const chips = el('div',null,'rec-tags');
    for (const x of data.plan.exclusions || []) chips.append(el('span',exclusions[x] || x));
    planBox.append(chips);
    if ((data.plan.warnings || []).length) planBox.append(el('p','용어와 제외 조건이 의도대로 해석됐는지 확인해 주세요.','live-rec-muted'));
    results.replaceChildren();
    for (const paper of data.results) {
      const card = el('article',null,'rec-paper');
      card.append(el('div',String(paper.rank).padStart(2,'0'),'rec-rank'),el('h4',paper.title));
      const tags = el('div',null,'rec-tags');
      for (const tag of [...(paper.tags.task || []),...(paper.tags.method || [])].slice(0,5)) tags.append(el('span',tag));
      card.append(tags);
      const count = Object.keys(paper.route_ranks || {}).length;
      card.append(el('p',`${count}개 검색 경로에서 발견된 논문입니다. 경로 중복이 조건 충족을 보장하지는 않습니다.`,'rec-reason'));
      const routeList = el('div',null,'rec-score-breakdown');
      for (const [key,name] of Object.entries(names)) routeList.append(el('span',`${name} · ${paper.route_ranks[key] ? paper.route_ranks[key]+'위' : '상위 후보에 없음'}`));
      card.append(routeList);
      if ((paper.missing_facets || []).length) card.append(el('p',`주제 표현 확인 필요: ${paper.missing_facets.map(x=>facets[x] || x).join(', ')}. 관련 후보이며 모든 조건이 확인된 것은 아닙니다.`,'live-rec-unknown'));
      const checks = Object.entries(paper.conditions || {});
      if (!checks.length) card.append(el('p','별도로 확인된 필수 조건은 없습니다. 초록에서 적합성을 확인해 주세요.','live-rec-muted'));
      for (const [key,value] of checks) {
        card.append(el('p',`${conditions[key] || key}: ${value === 'reported_in_abstract' ? '초록에 근거 표현 있음' : '확인 불가'}`,value === 'reported_in_abstract' ? 'live-rec-confirmed' : 'live-rec-unknown'));
        if (paper.evidence[key]) { const d=el('details');d.append(el('summary','근거 구절'),el('p',paper.evidence[key]));card.append(d); }
      }
      const abstract = el('details');abstract.append(el('summary','초록 읽기'),el('p',paper.abstract || '초록이 없습니다.'));card.append(abstract);
      card.append(externalLink('arXiv에서 제목으로 검색 ↗', paperSearch(paper.title), 'live-rec-link'));
      results.append(card);
    }
    const authors = data.recommended_authors || [];
    const hasPapers = data.results.length > 0;
    paperHeading.hidden = !hasPapers;
    authorSection.hidden = !hasPapers;
    authorJump.hidden = !authors.length;
    renderAuthors(authors);
    status.textContent = hasPapers ? `논문 ${data.results.length}편${authors.length ? ` · 저자 ${authors.length}명` : ''} · 검색 계산 ${(data.search_ms/1000).toFixed(2)}초 (초기 준비·대기 시간 제외)` : '해석한 조건을 뒷받침하는 후보를 찾지 못했습니다. 조건을 줄이거나 영어 전문 용어로 다시 검색해 보세요.';
  }
  async function json(url, options) {
    const response=await fetch(url,options);const data=await response.json();
    if (!response.ok) throw new Error(data.error || '검색 요청에 실패했습니다.');return data;
  }
  authorJump.addEventListener('click', () => {
    document.querySelector('#live-rec-authors-title').focus();
    authorSection.scrollIntoView({block: 'start'});
  });
  document.querySelectorAll('[data-rec-example]').forEach(b=>b.addEventListener('click',()=>{input.value=b.dataset.recExample;input.focus();}));
  form.addEventListener('submit',async event=>{
    event.preventDefault();if(button.disabled || !input.value.trim()) return;
    button.disabled=true;button.textContent='검색 중…';form.setAttribute('aria-busy','true');
    results.replaceChildren();authorResults.replaceChildren();planBox.hidden=true;paperHeading.hidden=true;authorSection.hidden=true;status.textContent='검색 요청을 보내고 있습니다…';status.classList.remove('live-rec-error');
    try {
      let job=await json('/api/cpu-recommend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query:input.value.trim()})});
      while (!['done','error'].includes(job.status)) {
        status.textContent=job.message;await new Promise(resolve=>setTimeout(resolve,job.status === 'loading' ? 1000 : 250));
        job=await json('/api/cpu-recommend/'+encodeURIComponent(job.id));
      }
      if(job.status==='error') throw new Error(job.message);render(job.result);
    } catch(error) {status.textContent=error.message;status.classList.add('live-rec-error');}
    finally {button.disabled=false;button.textContent='논문·저자 찾기 →';form.removeAttribute('aria-busy');}
  });
})();
