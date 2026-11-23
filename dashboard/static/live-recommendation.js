(() => {
  const form = document.querySelector('#live-rec-form');
  const input = document.querySelector('#live-rec-query');
  const submit = document.querySelector('#live-rec-submit');
  const history = document.querySelector('#live-rec-history');
  const chat = history.closest('.rec-chat');
  const welcome = document.querySelector('#live-rec-welcome');
  const announcement = document.querySelector('#live-rec-announcement');
  const filters = document.querySelector('#live-rec-filters');
  const examples = document.querySelector('#live-rec-examples');
  const followups = document.querySelector('#live-rec-followups');
  const queryLabel = document.querySelector('#live-rec-query-label');
  const reset = document.querySelector('#live-rec-reset');
  const exampleButtons = document.querySelectorAll('[data-rec-example]');
  const storageKey = 'arxiv-intelligence.recommendation-chat.v1';
  const historyLimit = 24;
  const clock = new Intl.DateTimeFormat('ko-KR', {hour: 'numeric', minute: '2-digit'});
  const citationNumber = new Intl.NumberFormat('ko-KR');
  const state = restore();
  let pendingSearch = null;

  const routeNames = {
    dtm: 'D/T/M', graph: '인용 그래프', text: '제목·초록',
    e5: '의미 검색', canonical: '정규화 의미 검색',
  };
  const facets = {
    node_classification: '노드 분류', graph_learning: '그래프 학습', fpga: 'FPGA',
    neural_network: '신경망', differential_privacy: '차등 프라이버시',
    collaborative_learning: '공동 학습', nonconvex: '비볼록',
    stochastic_gradient: '확률적 경사', implicit_feedback: '암묵적 피드백',
  };
  const conditions = {power: '전력 수치', throughput: '처리량', node_features: '노드 특성'};
  const exclusions = {survey: '서베이 제외', attack: '공격 연구 제외', convex_only: '볼록 전용 연구 제외'};
  const turnLabels = {initial: '첫 질문', followup: '후속 질문', remove: '조건 해제'};

  function restore() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(storageKey));
      if (Array.isArray(saved?.turns)) return saved;
    } catch {
      // The chat still works if browser storage is blocked.
    }
    return {turns: [], active: null};
  }

  function save() {
    try {
      sessionStorage.setItem(storageKey, JSON.stringify(state));
    } catch {
      // Keep the in-memory chat available when storage is full or blocked.
    }
  }

  function el(tag, text, className) {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (className) element.className = className;
    return element;
  }

  function action(text, className, onClick) {
    const button = el('button', text, className);
    button.type = 'button';
    button.addEventListener('click', onClick);
    return button;
  }

  function externalLink(label, url, className) {
    const link = el('a', label, className);
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    return link;
  }

  function paperSearch(title) {
    return 'https://arxiv.org/search/?searchtype=title&query=' + encodeURIComponent(title);
  }

  function disclosure(label, ...content) {
    const details = el('details');
    details.append(el('summary', label), ...content);
    return details;
  }

  function paperCard(paper) {
    const card = el('article', null, 'rec-paper');
    const citationLabel = paper.citation_scope === 'sample' ? '표본 내 인용' : 'OGBN-Arxiv 인용';
    const citations = Number.isSafeInteger(paper.citation_count) && paper.citation_count >= 0
      ? citationLabel + ' ' + citationNumber.format(paper.citation_count) + '회'
      : '인용 수 확인 불가';
    card.append(
      el('span', String(paper.rank).padStart(2, '0'), 'rec-rank'),
      el('h4', paper.title),
      el('p', citations, 'live-rec-muted'),
    );

    const tags = [...(paper.tags?.task || []), ...(paper.tags?.method || [])].slice(0, 5);
    if (tags.length) {
      const chips = el('div', null, 'rec-tags');
      chips.append(...tags.map(tag => el('span', tag)));
      card.append(chips);
    }

    const ranks = paper.route_ranks || {};
    const routeList = el('div', null, 'rec-score-breakdown');
    for (const [key, name] of Object.entries(routeNames)) {
      routeList.append(el('span', name + ' · ' + (ranks[key] ? ranks[key] + '위' : '상위 후보에 없음')));
    }

    const evidence = [
      el('p', Object.keys(ranks).length + '개 검색 경로에서 찾았습니다. 조건 충족 여부는 제목·초록의 근거를 확인해 주세요.'),
      routeList,
    ];
    if (paper.initial_evidence) {
      evidence.push(el('p', '첫 질문 근거'), el('blockquote', paper.initial_evidence));
    }
    if (paper.missing_facets?.length) {
      const terms = paper.missing_facets.map(key => facets[key] || key).join(', ');
      evidence.push(el('p', '주제 표현 확인 필요 · ' + terms, 'live-rec-unknown'));
    }
    const checks = Object.entries(paper.conditions || {});
    if (!checks.length && !paper.followup_evidence?.length) {
      evidence.push(el('p', '별도로 확인된 필수 조건이 없습니다. 초록에서 적합성을 확인해 주세요.'));
    }
    for (const [key, value] of checks) {
      const confirmed = value === 'reported_in_abstract';
      evidence.push(el(
        'p',
        (conditions[key] || key) + ' · ' + (confirmed ? '초록에 근거 표현 있음' : '확인 불가'),
        confirmed ? 'live-rec-confirmed' : 'live-rec-unknown',
      ));
      if (paper.evidence?.[key]) evidence.push(el('blockquote', paper.evidence[key]));
    }
    for (const item of paper.followup_evidence || []) {
      evidence.push(el('p', '후속 조건 근거 · ' + item.label), el('blockquote', item.quote));
    }

    card.append(
      disclosure('검색 경로 및 조건 근거', ...evidence),
      disclosure('초록 읽기', el('p', paper.abstract || '초록이 없습니다.')),
      externalLink('arXiv에서 제목으로 검색 ↗', paperSearch(paper.title), 'live-rec-link'),
    );
    return card;
  }

  function authorCard(author) {
    const card = el('article', null, 'rec-author');
    card.append(
      el('h4', author.author_name),
      el('p', '상위 후보의 근거 논문 ' + author.related_paper_count + '편'),
    );
    const evidence = el('ul');
    for (const paper of author.evidence_papers) {
      const row = el('li');
      row.append(paper.title
        ? externalLink(paper.title, paperSearch(paper.title))
        : document.createTextNode('논문 ' + paper.paper_id));
      evidence.append(row);
    }
    if (evidence.children.length) card.append(disclosure('근거 논문 보기', evidence));
    if (/^https:\/\/openalex\.org\/A\d+$/.test(author.author_id)) {
      card.append(externalLink('OpenAlex 프로필 ↗', author.author_id, 'live-rec-link'));
    }
    return card;
  }

  function searchDetails({conversation, plan, search_ms}) {
    const details = disclosure(
      '탐색 조건 및 검색 정보',
      el('p', '탐색 주제 · ' + conversation.topic),
      el('p', '해석한 검색어 · ' + (plan.positive_query || '인식된 검색어 없음')),
      el('p', Number.isInteger(conversation.initial_verified_candidate_count)
        ? '첫 검색 후보 ' + conversation.initial_candidate_count + '편 검토 · 근거 확인 ' + conversation.initial_verified_candidate_count + '편'
        : '첫 검색 후보 ' + conversation.initial_candidate_count + '편'),
      ...(Number.isInteger(conversation.reviewed_candidate_count)
        ? [el('p', '후속 조건 관련 후보 ' + conversation.reviewed_candidate_count + '편 검토 · 근거 확인 ' + conversation.candidate_count + '편')]
        : []),
      el('p', '검색 계산 ' + search_ms + 'ms (초기 준비·대기 시간 제외)'),
    );
    details.className = 'rec-search-details';
    const labels = [
      ...(plan.exclusions || []).map(key => exclusions[key] || key),
      ...conversation.filters.map(filter => filter.label),
    ];
    if (labels.length) {
      const chips = el('div', null, 'rec-tags');
      chips.append(...labels.map(label => el('span', label)));
      details.append(chips);
    }
    if (plan.warnings?.length) {
      details.append(el('p', '용어와 제외 조건이 의도대로 해석됐는지 확인해 주세요.'));
    }
    return details;
  }

  function renderAnswer(container, data) {
    const papers = data.results;
    const authors = data.recommended_authors;
    if (!papers.length) {
      const message = data.conversation.filters.length
        ? '검토한 논문의 제목과 초록에서 현재 조건을 뒷받침하는 근거를 찾지 못했어요. 적용 중인 조건을 해제하거나 새 탐색을 시작할 수 있어요.'
        : data.conversation.initial_candidate_count > 0 && data.conversation.initial_verified_candidate_count === 0
          ? '검토한 논문 ' + data.conversation.initial_candidate_count + '편의 제목과 초록에서 질문의 필수 조건을 확인하지 못했어요.'
          : '질문에 맞는 논문을 찾지 못했어요. 다른 검색어로 다시 질문해 주세요.';
      container.replaceChildren(el('p', message), searchDetails(data));
      return message;
    }

    const message = '관련 논문 ' + papers.length + '편' +
      (authors.length ? '과 추천 저자 ' + authors.length + '명을 찾았어요.' : '을 찾았어요.');
    const paperHeading = el('div', null, 'rec-section-title');
    paperHeading.append(el('h3', '추천 논문'));
    const paperResults = el('div', null, 'rec-paper-grid');
    paperResults.append(...papers.map(paperCard));

    const authorHeading = el('h3', '추천 저자', 'rec-section-title');
    const authorResults = el('div', null, 'rec-author-grid');
    if (authors.length) {
      authorResults.append(...authors.map(authorCard));
      authorHeading.tabIndex = -1;
      paperHeading.append(action('추천 저자 보기 ↓', 'live-rec-jump', () => {
        authorHeading.focus({preventScroll: true});
        scrollToMessage(authorHeading);
      }));
    } else {
      authorResults.append(el('p', '검색 후보에서 저자 정보를 확인하지 못했어요.', 'live-rec-muted'));
    }

    container.replaceChildren(
      el('p', message),
      searchDetails(data),
      paperHeading,
      paperResults,
      authorHeading,
      ...(authors.length ? [el('p', '조건이 확인된 후보 논문에 많이 등장한 저자부터 보여줍니다.', 'live-rec-muted')] : []),
      authorResults,
    );
    return message;
  }

  function appendTurn(turn) {
    const node = el('div', null, 'rec-turn');
    const question = el('div', null, 'rec-user');
    const meta = el('div', null, 'rec-user-meta');
    const time = el('time', clock.format(turn.timestamp));
    time.dateTime = new Date(turn.timestamp).toISOString();
    meta.append(el('strong', '나 · ' + turnLabels[turn.kind]), time);
    question.append(meta, el('p', turn.prompt));

    const answer = el('div', null, 'rec-assistant');
    const avatar = el('span', 'AI', 'rec-ai-mark');
    avatar.setAttribute('aria-hidden', 'true');
    const bubble = el('div', null, 'rec-answer');
    const content = el('div', null, 'rec-answer-content');
    bubble.append(el('strong', '논문 탐색 도우미', 'rec-speaker'), content);
    answer.append(avatar, bubble);
    node.append(question, answer);
    history.append(node);

    if (turn.result) {
      renderAnswer(content, turn.result);
    } else if (turn.clarification) {
      renderClarification(content, turn.clarification);
    } else {
      showStatus(content, turn.error || turn.status, Boolean(turn.error));
    }
    return {node, content};
  }

  function scrollToMessage(element) {
    if (!history.clientHeight) return;
    const top = element.getBoundingClientRect().top - history.getBoundingClientRect().top;
    history.scrollTop += top - 12;
  }

  function showStatus(container, message, error = false) {
    const status = el('p', message, error ? 'live-rec-error' : 'rec-progress');
    container.replaceChildren(status);
  }

  function renderClarification(container, {reason, question}) {
    container.replaceChildren(el('p', reason), el('p', question));
  }

  function updateComposer() {
    const busy = Boolean(pendingSearch);
    const active = state.active;
    const awaitingAnswer = active?.awaitingAnswer;
    examples.hidden = Boolean(active);
    followups.hidden = !active;
    reset.disabled = state.turns.length === 0;
    submit.disabled = busy;
    input.readOnly = busy;
    form.setAttribute('aria-busy', String(busy));
    submit.textContent = busy ? '검색 중…' : awaitingAnswer ? '답변 보내기' : active ? '조건 적용' : '검색 시작';
    exampleButtons.forEach(button => { button.disabled = busy; });
    queryLabel.textContent = awaitingAnswer ? '확인 질문에 답하기' : active ? '후속 질문' : '첫 질문';
    input.placeholder = awaitingAnswer ? '답변을 입력해 주세요.' : active
      ? '예: 실험에서 쓴 전력이 5와트를 넘지 않는 논문으로 좁혀줘.'
      : '예: FPGA 신경망 추론 구현 논문을 찾아줘.';

    const activeFilters = active?.filters || [];
    filters.hidden = !activeFilters.length;
    filters.replaceChildren();
    for (const {id, label} of activeFilters) {
      const chip = action(label + ' ×', 'live-rec-filter', () => {
        input.focus({preventScroll: true});
        search({remove_filter: id}, 'remove', label);
      });
      chip.disabled = busy;
      chip.setAttribute('aria-label', label + ' 조건 해제');
      filters.append(chip);
    }
  }

  async function json(url, options) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => null);
    if (!response.ok || !data) {
      throw new Error(data?.error || '검색 서버의 응답을 받지 못했어요. 잠시 후 다시 시도해 주세요.');
    }
    return data;
  }

  async function runTurn(turn, content, payload) {
    const request = new AbortController();
    const {signal} = request;
    pendingSearch = request;
    chat.dataset.searchVisual = 'searching';
    updateComposer();

    try {
      let job;
      if (payload) {
        job = await json('/api/cpu-recommend', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify(payload),
          signal,
        });
        signal.throwIfAborted();
        turn.jobId = job.id;
        save();
      } else if (turn.jobId) {
        job = await json('/api/cpu-recommend/' + encodeURIComponent(turn.jobId), {signal});
      } else {
        throw new Error('새로고침으로 검색 요청 확인이 중단됐어요. 질문을 다시 보내 주세요.');
      }

      while (!['done', 'clarify', 'error'].includes(job.status)) {
        signal.throwIfAborted();
        const status = job.message || '검색 중입니다.';
        if (status !== turn.status) {
          turn.status = status;
          showStatus(content, status);
          announcement.textContent = status;
        }
        await new Promise(resolve => setTimeout(resolve, job.status === 'loading' ? 1000 : 350));
        signal.throwIfAborted();
        job = await json('/api/cpu-recommend/' + encodeURIComponent(turn.jobId), {signal});
      }
      signal.throwIfAborted();
      if (job.status === 'error') throw new Error(job.message);
      if (job.status === 'clarify') {
        turn.clarification = job.clarification;
        state.active = {...state.active, id: job.id, awaitingAnswer: true};
        renderClarification(content, turn.clarification);
        announcement.textContent = job.message;
        return;
      }

      const {conversation, plan, results, recommended_authors = [], search_ms} = job.result;
      const data = {conversation, plan, results, recommended_authors, search_ms};
      announcement.textContent = renderAnswer(content, data);
      turn.result = data;
      state.active = {id: job.id, filters: conversation.filters};
    } catch (error) {
      if (signal.aborted) return;
      turn.error = error.message;
      showStatus(content, turn.error, true);
      announcement.textContent = turn.error;
      if (payload?.query) input.value = turn.prompt;
    } finally {
      if (pendingSearch === request) {
        pendingSearch = null;
        if (turn.result || turn.clarification) {
          chat.dataset.searchVisual = 'complete';
        } else {
          delete chat.dataset.searchVisual;
        }
        updateComposer();
        save();
        if (payload && turn.clarification) input.focus({preventScroll: true});
      }
    }
  }

  function search(payload, kind, prompt) {
    if (pendingSearch) return;
    if (state.active) payload.previous_id = state.active.id;

    const turn = {kind, prompt, timestamp: Date.now(), status: '검색 요청을 보내고 있어요…'};
    if (state.turns.length >= historyLimit) {
      state.turns.shift();
      history.querySelector('.rec-turn').remove();
    }
    state.turns.push(turn);
    const {node, content} = appendTurn(turn);
    history.scrollTop = history.scrollHeight;
    announcement.textContent = turn.status;
    save();
    if (payload.query) input.value = '';
    if (node.offsetHeight > history.clientHeight) scrollToMessage(node);
    runTurn(turn, content, payload);
  }

  reset.addEventListener('click', () => {
    pendingSearch?.abort();
    pendingSearch = null;
    delete chat.dataset.searchVisual;
    state.turns.length = 0;
    state.active = null;
    history.replaceChildren(welcome);
    history.scrollTop = 0;
    input.value = '';
    announcement.textContent = '새 탐색을 시작합니다.';
    updateComposer();
    save();
    input.focus();
  });

  for (const example of exampleButtons) {
    example.addEventListener('click', () => {
      if (pendingSearch) return;
      input.value = example.dataset.recExample;
      input.focus();
    });
  }

  form.addEventListener('submit', event => {
    event.preventDefault();
    const query = input.value.trim();
    if (query) search({query}, state.active ? 'followup' : 'initial', query);
  });

  input.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing && event.keyCode !== 229) {
      event.preventDefault();
      if (!pendingSearch) form.requestSubmit();
    }
  });

  let latest;
  for (const turn of state.turns) latest = appendTurn(turn);
  updateComposer();
  if (latest) {
    const showLatest = () => {
      if (!history.clientHeight) return;
      if (latest.node.parentNode === history) scrollToMessage(latest.node);
      document.removeEventListener('dashboard:navigate', showLatest);
    };
    if (history.clientHeight) requestAnimationFrame(showLatest);
    else document.addEventListener('dashboard:navigate', showLatest);

    const turn = state.turns.at(-1);
    if (!turn.result && !turn.clarification && !turn.error) runTurn(turn, latest.content);
  }
})();
