// Presentation-only fixtures. No recommendation or paid API requests are made.
(() => {
  const papers = [
    { title: 'Self-supervised Graph Learning for Recommendation', authors: 'Jiancan Wu et al.', year: '2021', venue: 'SIGIR', tags: ['자기지도 학습', '그래프 추천'], reason: '그래프 증강과 대조 학습으로 사용자–아이템 관계를 학습하는 연구예요.', text: 0.842, intent: 0.026, dtm: 0.024 },
    { title: 'LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation', authors: 'Xiangnan He et al.', year: '2020', venue: 'SIGIR', tags: ['LightGCN', '협업 필터링'], reason: '간결한 그래프 전파 구조로 추천 성능을 높여, 비교 기준으로 살펴보기 좋아요.', text: 0.831, intent: 0.020, dtm: 0.016 },
    { title: 'Neural Graph Collaborative Filtering', authors: 'Xiang Wang et al.', year: '2019', venue: 'SIGIR', tags: ['그래프 신경망', '고차 연결'], reason: '사용자와 아이템의 고차 연결을 반영하는 그래프 기반 추천의 기초 연구예요.', text: 0.805, intent: 0.014, dtm: 0.012 },
  ];
  const authors = [
    { name: 'Jiancan Wu', initials: 'JW', area: '자기지도 학습 · 그래프 추천', max: 0.892, mean: 0.826, count: 6 },
    { name: 'Xiangnan He', initials: 'XH', area: '협업 필터링 · 표현 학습', max: 0.867, mean: 0.796, count: 5 },
    { name: 'Xiang Wang', initials: 'XW', area: '그래프 신경망 · 추천 시스템', max: 0.831, mean: 0.762, count: 4 },
  ];
  const number = (value) => value.toFixed(3);
  document.querySelector('#rec-question').textContent = '그래프 신경망을 활용한 추천 시스템 논문을 찾고 있어. 대조 학습을 적용한 연구와 관련 연구자도 추천해줘.';
  document.querySelector('#rec-papers').innerHTML = papers.map((paper, index) => {
    const score = paper.text + paper.intent + paper.dtm;
    return `<article class="rec-paper"><div class="rec-card-top"><span class="rec-rank">0${index + 1}</span><span class="rec-venue">${paper.venue} · ${paper.year}</span><span class="rec-match">추천 점수 <b>${number(score)}</b></span></div><h4>${paper.title}</h4><p class="rec-paper-byline">${paper.authors}</p><div class="rec-tags">${paper.tags.map(tag => `<span>${tag}</span>`).join('')}</div><p class="rec-reason">${paper.reason}</p><div class="rec-score-breakdown"><span>텍스트 유사도 <b>${number(paper.text)}</b></span><span>의도 보정 <b>+${number(paper.intent)}</b></span><span>DTM 가중 기여 <b>+${number(paper.dtm)}</b></span><span>그래프 감점 <b>−0.000</b></span></div></article>`;
  }).join('');
  document.querySelector('#rec-authors').innerHTML = authors.map((author, index) => {
    const countScore = Math.log1p(author.count) / Math.log1p(10);
    const score = 0.4 * author.max + 0.4 * author.mean + 0.2 * countScore;
    return `<article class="rec-author"><div class="rec-author-top"><span class="rec-author-avatar">${author.initials}</span><div><h4>${author.name}</h4><p>${author.area}</p></div><div class="rec-author-total"><small>추천 점수</small><b>${number(score)}</b></div></div><div class="rec-author-values"><span>최대 관련도<b>${number(author.max)}</b></span><span>평균 관련도<b>${number(author.mean)}</b></span><span>논문 수 점수<b>${number(countScore)}</b></span></div><div class="rec-author-bottom"><span>관련 논문 <b>${author.count}편</b></span><span>추천 연구자 0${index + 1}</span></div></article>`;
  }).join('');
  document.querySelector('#rec-form').addEventListener('submit', event => {
    event.preventDefault();
    const input = document.querySelector('#rec-input');
    if (!input.value.trim()) return;
    document.querySelector('#rec-question').textContent = input.value.trim();
    document.querySelector('#rec-answer').textContent = '입력하신 질문을 반영한 대화 화면 예시입니다. 아래 논문과 연구자, 점수는 발표용으로 준비된 동일한 예시 데이터입니다.';
    input.value = '';
  });
})();
