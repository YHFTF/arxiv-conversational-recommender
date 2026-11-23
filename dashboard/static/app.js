// ============================================================
// DOM Helpers
// ============================================================

const $ = (selector, root = document) => root.querySelector(selector);

const $$ = (selector, root = document) => [
  ...root.querySelectorAll(selector),
];


// ============================================================
// Constants
// ============================================================

const COLAB_URL =
  'https://colab.research.google.com/github/YHFTF/arxiv-conversational-recommender/blob/main/notebooks/GPU_Workbench.ipynb';

const PAGE_NAMES = {
  home: '개요',
  project: '프로젝트',
  scripts: '스크립트 실행',
  learn: '모델 학습',
  benchmark: '벤치마크',
  production: '논문 탐색',
  'data-structure': '데이터 구조',
};

const JOB_STATUS_TEXT = {
  queued: '대기 중',
  running: '실행 중',
  success: '완료',
  failed: '실패',
  stopping: '중지 중',
  cancelled: '중지됨',
};


// ============================================================
// Application State
// ============================================================

let gitState = {
  branches: [],
};

let issues = [];

let currentJobId = null;
let jobPollTimer = null;
let toastTimer;
let isAdmin = false;
let hasLivePaperStructure = false;


function setButtonBusy(button, busy, label = '') {
  if (!button) return;

  if (busy) {
    button.dataset.idleLabel ||= button.textContent.trim();
    button.textContent = label || button.dataset.idleLabel;
    button.classList.add('is-busy');
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    return;
  }

  button.textContent = button.dataset.idleLabel || button.textContent;
  button.classList.remove('is-busy');
  button.disabled = false;
  button.removeAttribute('aria-busy');
}


// ============================================================
// API
// ============================================================

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      'X-Dashboard-Request': '1',
    },
    ...options,
  });

  const data = await response.json();

  if (!response.ok) {
    if (response.status === 401 && isAdmin) {
      window.location.reload();
    }
    const error = new Error(
      data.error ||
      data.output ||
      '요청 실패',
    );
    error.data = data;
    throw error;
  }

  return data;
}


// ============================================================
// Utilities
// ============================================================

function escapeHtml(value = '') {
  const entities = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  };

  return String(value).replace(
    /[&<>"']/g,
    (char) => entities[char],
  );
}


function formatRelativeTime(date) {
  const seconds =
    (Date.now() - new Date(date).getTime()) / 1000;

  if (seconds < 60) {
    return '방금 전';
  }

  if (seconds < 3600) {
    return `${Math.floor(seconds / 60)}분 전`;
  }

  if (seconds < 86400) {
    return `${Math.floor(seconds / 3600)}시간 전`;
  }

  return `${Math.floor(seconds / 86400)}일 전`;
}


function showToast(message, isError = false) {
  const toast = $('#toast');

  toast.textContent = message;
  toast.classList.toggle('toast-error', isError);

  toast.classList.add('show');

  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.classList.remove('show');
  }, 2800);
}


// ============================================================
// Navigation
// ============================================================

function navigate(pageId) {
  if (!isAdmin && pageId !== 'data-structure') pageId = 'production';
  $$('.page').forEach((page) => {
    page.classList.toggle(
      'active',
      page.id === pageId,
    );
  });

  $$('aside nav button').forEach((button) => {
    const active = button.dataset.page === pageId;
    button.classList.toggle('active', active);
    if (active) {
      button.setAttribute('aria-current', 'page');
    } else {
      button.removeAttribute('aria-current');
    }
  });

  $('#title').textContent = PAGE_NAMES[pageId];

  if (pageId === 'project') {
    loadDocs();
  }
  if (pageId === 'data-structure') {
    loadPaperStructure();
  }
  document.dispatchEvent(new Event('dashboard:navigate'));
}


function initNavigation() {
  $$('aside nav button, [data-destination], .brand').forEach((control) => {
    control.addEventListener('click', (event) => {
      event.preventDefault();
      const pageId = control.dataset.page || control.dataset.destination || (isAdmin ? 'home' : 'production');
      if (location.hash !== `#${pageId}`) {
        history.pushState(null, '', `#${pageId}`);
      }
      navigate(pageId);
      window.scrollTo(0, 0);
      const heading = $(`#${pageId} h1`);
      heading.tabIndex = -1;
      heading.focus({ preventScroll: true });
    });
  });

  $('.skip-link').addEventListener('click', (event) => {
    event.preventDefault();
    $('#dashboard-content').focus();
  });

  const pageFromLocation = () => {
    const pageId = location.hash.slice(1);
    return Object.hasOwn(PAGE_NAMES, pageId) ? pageId : (isAdmin ? 'home' : 'production');
  };
  navigate(pageFromLocation());
  window.addEventListener('hashchange', () => {
    navigate(pageFromLocation());
    window.scrollTo(0, 0);
  });
}


function initTheme() {
  const toggle = $('#theme-toggle');
  const root = document.documentElement;
  try {
    if (localStorage.getItem('arxiv-dashboard-theme') === 'dark') {
      root.dataset.theme = 'dark';
    }
  } catch {
    // The default theme also works when browser storage is unavailable.
  }
  const updateToggle = () => {
    const dark = root.dataset.theme === 'dark';
    toggle.setAttribute('aria-pressed', String(dark));
    toggle.title = dark ? '기본 테마로 전환' : '다크 모드로 전환';
  };

  updateToggle();
  toggle.addEventListener('click', () => {
    root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark';
    updateToggle();
    try {
      localStorage.setItem('arxiv-dashboard-theme', root.dataset.theme);
    } catch {
      // The theme still switches when browser storage is unavailable.
    }
  });
}

async function loadPaperStructure() {
  const status = $('#paper-schema-status');
  if (status.getAttribute('aria-busy') === 'true') return;

  const retry = $('#paper-schema-retry');
  status.setAttribute('aria-busy', 'true');
  retry.hidden = true;
  try {
    // Older servers may return a 401 for this route even with a valid session.
    const response = await fetch('/api/data-structure', {signal: AbortSignal.timeout(4000)});
    if (!response.ok) throw new Error('Data structure API unavailable');
    renderPaperStructure(await response.json());
    hasLivePaperStructure = true;
    status.hidden = true;
  } catch {
    if (hasLivePaperStructure) {
      status.textContent = '최신 집계를 불러오지 못했습니다. 이전 집계를 표시합니다.';
    } else {
      const snapshot = JSON.parse($('#paper-dataset-snapshot').textContent);
      renderPaperStructure(snapshot);
      status.textContent = `${snapshot.asOf} 기준 집계 · 서버 집계에 연결되면 갱신됩니다.`;
    }
    status.hidden = false;
    retry.hidden = false;
  } finally {
    status.removeAttribute('aria-busy');
  }
}

function renderPaperStructure(data) {
  const format = new Intl.NumberFormat('ko-KR');
  const counts = {
    '#paper-schema-total': `${format.format(data.papers)}편`,
    '#paper-schema-authors': `${format.format(data.authorEdges)}개 연결`,
    '#paper-schema-citations': `${format.format(data.citationEdges)}개 연결`,
    '#paper-schema-topics': `${format.format(data.categories.length)}개 분야`,
  };
  for (const [selector, label] of Object.entries(counts)) {
    const element = $(selector);
    element.textContent = label;
    element.hidden = false;
  }

  const source = $('#paper-schema-source-counts');
  const sourceCounts = [];
  if (Number.isInteger(data.source.papers)) {
    sourceCounts.push(`논문 ${format.format(data.source.papers)}편`);
  }
  if (Number.isInteger(data.source.citationEdges)) {
    sourceCounts.push(`인용 ${format.format(data.source.citationEdges)}개 연결`);
  }
  source.textContent = sourceCounts.join(' · ');
  source.hidden = !sourceCounts.length;
  $('#paper-schema-dtm-count').textContent = ` · ${format.format(data.dtmSuccess)}편 추출 완료`;

  const maxCount = Math.max(1, ...data.categories.map((category) => category.papers));
  $('#paper-distribution-chart').replaceChildren(...data.categories.map((category) => {
    const row = document.createElement('li');
    const label = document.createElement('span');
    label.textContent = category.code || `분야 ${category.id}`;
    const bar = document.createElement('span');
    bar.className = 'paper-distribution-bar';
    bar.style.setProperty('--bar-width', `${category.papers / maxCount * 100}%`);
    bar.setAttribute('aria-hidden', 'true');
    const count = document.createElement('span');
    count.textContent = `${format.format(category.papers)}편`;
    row.append(label, bar, count);
    return row;
  }));
  $('#paper-distribution-total').textContent = `${format.format(data.categories.length)}개 분야`;
  $('#paper-distribution').hidden = false;
}


// ============================================================
// Colab Links
// ============================================================

function initColabLinks() {
  $$('[data-colab]').forEach((link) => {
    link.href = COLAB_URL;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
  });
}


// ============================================================
// Git
// ============================================================

async function loadGit() {
  try {
    gitState = await api('/api/git');

    $('#branch-pill').textContent = gitState.branch;
    $('#branch-pill').title = gitState.branch;

    $('#hero-branch').textContent =
      gitState.branch;

    $('#repo-state').textContent =
      gitState.dirty
        ? `미커밋 변경 ${gitState.changes}개`
        : '변경 없음';
    $('#repo-state').dataset.state = gitState.dirty ? 'dirty' : 'clean';
    $('#change-count').textContent = gitState.changes;

    $('#commit-count').textContent =
      gitState.commits.length;

    $('#branch-count').textContent =
      gitState.branches.length;

    renderCommits(gitState.commits);
  } catch (error) {
    $('#repo-state').textContent = '조회 실패';
    $('#repo-state').dataset.state = 'dirty';
    showToast(error.message, true);
  }
}


function renderCommits(commits) {
  const container = $('#commits');

  if (!commits.length) {
    container.innerHTML =
      '<small>커밋 없음</small>';
    return;
  }

  container.innerHTML = commits
    .slice(0, 10)
    .map((commit) => {
      const initials = escapeHtml(
        commit.author
          .slice(0, 2)
          .toUpperCase(),
      );

      return `
        <div class="commit">
          <i>${initials}</i>

          <div>
            <b>${escapeHtml(commit.message)}</b>

            <small>
              ${escapeHtml(commit.author)}
              ·
              ${formatRelativeTime(commit.date)}
            </small>
          </div>

          <code>
            ${escapeHtml(commit.hash)}
          </code>
        </div>
      `;
    })
    .join('');
}


async function pullGit() {
  const button = $('#pull');

  button.disabled = true;

  try {
    const result = await api(
      '/api/git/pull',
      {
        method: 'POST',
        body: JSON.stringify({ branch: gitState.branch }),
      },
    );

    showToast(
      result.output ||
      '최신 상태입니다.',
    );

    loadGit();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
  }
}


function openBranchModal() {
  $('#branch-list').innerHTML =
    gitState.branches
      .map(
        (branch) => `
          <div class="branch-row">
            <button data-branch-history="${escapeHtml(branch.name)}"><code>${escapeHtml(branch.name)}</code></button>

            <small>
              ${escapeHtml(branch.hash)}
              ·
              ${escapeHtml(branch.message)}
            </small>
            <button data-branch-switch="${escapeHtml(branch.name)}">전환</button>
          </div>
        `,
      )
      .join('');

  $$('[data-branch-history]').forEach((button) => {
    button.onclick = () => loadBranchHistory(button.dataset.branchHistory);
  });

  $$('[data-branch-switch]').forEach((button) => {
    button.onclick = () => changeBranch(button.dataset.branchSwitch);
  });

  $('#modal').showModal();

  loadBranchHistory(gitState.branch);
}


async function loadBranchHistory(branch) {
  try {
    const data = await api(`/api/git/commits?branch=${encodeURIComponent(branch)}`);
    $('#branch-history-title').textContent = `${branch} 개발 내역`;
    const container = $('#branch-history');
    container.innerHTML = '';
    data.commits.forEach((commit) => {
      const row = document.createElement('div');
      row.className = 'branch-commit';
      row.innerHTML = `<b>${escapeHtml(commit.message)}</b><small>${escapeHtml(commit.author)} · ${formatRelativeTime(commit.date)}</small><code>${escapeHtml(commit.hash)}</code>`;
      container.appendChild(row);
    });
    if (!data.commits.length) container.textContent = '커밋이 없습니다.';
  } catch (error) {
    showToast(error.message, true);
  }
}


async function changeBranch(branch) {
  try {
    const result = await api('/api/git/switch', {
      method: 'POST',
      body: JSON.stringify({ branch }),
    });
    showToast(result.output || `${result.branch}(으)로 전환했습니다.`);
    $('#modal').close();
    refreshAll();
  } catch (error) {
    showToast(error.message, true);
  }
}


// ============================================================
// GitHub Issues
// ============================================================

async function loadIssues(filter = 'all') {
  try {
    const data = await api('/api/issues');

    issues = data.issues;

    $('#issue-repo').textContent =
      data.repo || 'GitHub';

    $('#issue-count').textContent =
      data.error ? '—' : issues.length;

    renderIssueFilters(filter);
    renderIssues(filter);

    if (data.error) {
      $('#issues').innerHTML = `
        <small>
          ${escapeHtml(data.error)}
        </small>
      `;
    }
  } catch (error) {
    $('#issues').innerHTML = `
      <small>
        ${escapeHtml(error.message)}
      </small>
    `;
  }
}


function getIssueCounts() {
  const counts = {
    all: issues.length,
    critical: 0,
    high: 0,
    medium: 0,
    low: 0,
  };

  issues.forEach((issue) => {
    if (counts[issue.severity] !== undefined) {
      counts[issue.severity]++;
    }
  });

  return counts;
}


function renderIssueFilters(activeFilter) {
  const counts = getIssueCounts();

  $('#filters').innerHTML =
    Object.entries(counts)
      .map(([filter, count]) => `
        <button
          data-f="${filter}"
          class="${filter === activeFilter ? 'active' : ''}"
          aria-pressed="${filter === activeFilter}"
        >
          ${filter.toUpperCase()}
          ${count}
        </button>
      `)
      .join('');

  $$('#filters button').forEach((button) => {
    button.onclick = () => {
      renderIssues(button.dataset.f);
    };
  });
}


function renderIssues(filter) {
  $$('#filters button').forEach((button) => {
    const active = button.dataset.f === filter;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  });

  const filteredIssues = issues.filter(
    (issue) =>
      filter === 'all' ||
      issue.severity === filter,
  );

  if (!filteredIssues.length) {
    $('#issues').innerHTML =
      '<small>열린 이슈가 없습니다.</small>';
    return;
  }

  $('#issues').innerHTML =
    filteredIssues
      .map(
        (issue) => `
          <a
            class="issue ${escapeHtml(issue.severity)}"
            href="${escapeHtml(issue.url)}"
            target="_blank"
            rel="noopener noreferrer"
          >
            <i></i>

            <div>
              <b>
                ${escapeHtml(issue.title)}
              </b>

              <small>
                ${
                  escapeHtml(
                    issue.labels.join(' · ') ||
                    '라벨 없음',
                  )
                }
              </small>
            </div>

            <code>
              #${issue.number}
            </code>
          </a>
        `,
      )
      .join('');
}


// ============================================================
// Notes
// ============================================================

async function loadNotes() {
  try {
    const notes = await api('/api/notes');

    if (!notes.length) {
      $('#note-list').innerHTML =
        '<small>첫 메모를 남겨보세요.</small>';
      return;
    }

    $('#note-list').innerHTML =
      notes
        .map(
          (note) => `
            <div class="note">
              <b>
                ${escapeHtml(note.author)}
              </b>

              <time>
                ${formatRelativeTime(note.created_at)}
              </time>

              <p>
                ${escapeHtml(note.content)}
              </p>
            </div>
          `,
        )
        .join('');
  } catch (error) {
    showToast(error.message, true);
  }
}


async function submitNote(event) {
  event.preventDefault();

  const author = $('#author').value;
  const content = $('#content').value;
  const status = $('#note-status');
  const button = $('button', event.currentTarget);
  if (button.disabled) return;
  setButtonBusy(button, true, '등록 중…');
  status.textContent = '';
  status.dataset.error = 'false';

  try {
    await api('/api/notes', {
      method: 'POST',
      body: JSON.stringify({
        author,
        content,
      }),
    });

    try {
      localStorage.setItem('author', author);
    } catch {
      // Memo submission does not require browser storage.
    }

    $('#content').value = '';

    loadNotes();
    status.textContent = '메모를 등록했습니다.';
  } catch (error) {
    status.dataset.error = 'true';
    status.textContent = error.message;
  } finally {
    setButtonBusy(button, false);
  }
}


function initNotes() {
  try {
    $('#author').value = localStorage.getItem('author') || '';
  } catch {
    // The author can still be entered by hand.
  }

  $('#note-form').onsubmit =
    submitNote;
}


// ============================================================
// Markdown Renderer
// ============================================================

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(
      /`([^`]+)`/g,
      '<code>$1</code>',
    )
    .replace(
      /\*\*([^*]+)\*\*/g,
      '<strong>$1</strong>',
    )
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>',
    );
}


function renderMarkdown(source) {
  let html = '';

  let insideCodeBlock = false;
  let insideList = false;

  const closeList = () => {
    if (!insideList) {
      return;
    }

    html += '</ul>';
    insideList = false;
  };

  for (const line of source.split('\n')) {
    // Code fence
    if (line.startsWith('```')) {
      closeList();

      if (insideCodeBlock) {
        html += '</code></pre>';
      } else {
        html += '<pre><code>';
      }

      insideCodeBlock =
        !insideCodeBlock;

      continue;
    }

    // Contents of code block
    if (insideCodeBlock) {
      html += `${escapeHtml(line)}\n`;
      continue;
    }

    // Heading
    const heading =
      line.match(/^(#{1,4})\s+(.*)/);

    if (heading) {
      closeList();

      const level =
        heading[1].length;

      html += `
        <h${level}>
          ${renderInlineMarkdown(heading[2])}
        </h${level}>
      `;

      continue;
    }

    // List item
    const listItem =
      line.match(/^\s*[-*]\s+(.*)/);

    if (listItem) {
      if (!insideList) {
        html += '<ul>';
        insideList = true;
      }

      html += `
        <li>
          ${renderInlineMarkdown(listItem[1])}
        </li>
      `;

      continue;
    }

    closeList();

    // Paragraph
    if (line.trim()) {
      html += `
        <p>
          ${renderInlineMarkdown(line)}
        </p>
      `;
    }
  }

  closeList();

  if (insideCodeBlock) {
    html += '</code></pre>';
  }

  return html;
}


// ============================================================
// Documentation
// ============================================================

async function loadDocs() {
  const docList = $('#doc-list');

  // 이미 불러온 경우 다시 요청하지 않음
  if (docList.children.length) {
    return;
  }

  try {
    const data = await api('/api/docs');

    renderDocumentList(data.documents);
    renderDocumentTree(data.tree);

    if (data.documents.length) {
      openDoc($('.doc'));
    }
  } catch (error) {
    showToast(error.message, true);
  }
}


function renderDocumentList(documents) {
  $('#doc-list').innerHTML =
    documents
      .map(
        (document, index) => `
          <button
            class="doc ${index === 0 ? 'active' : ''}"
            data-p="${escapeHtml(document.path)}"
          >
            ${escapeHtml(document.name)}
          </button>
        `,
      )
      .join('');

  $$('.doc').forEach((button) => {
    button.onclick = () => {
      openDoc(button);
    };
  });
}


function renderDocumentTree(tree) {
  $('#tree').innerHTML =
    tree
      .map(
        (item) => `
          <div class="tree">
            ▾ ${escapeHtml(item.path)}
            <b>${item.files}</b>
          </div>
        `,
      )
      .join('');
}


async function openDoc(button) {
  $$('.doc').forEach((item) => {
    item.classList.toggle(
      'active',
      item === button,
    );
  });

  try {
    const path =
      encodeURIComponent(
        button.dataset.p,
      );

    const data = await api(
      `/api/doc?path=${path}`,
    );

    $('#markdown').innerHTML =
      renderMarkdown(data.content);
  } catch (error) {
    showToast(error.message, true);
  }
}


// ============================================================
// Tasks / Jobs
// ============================================================

async function loadTasks() {
  try {
    const data = await api('/api/tasks');

    Object.entries(data.tasks).forEach(([taskId, task]) => {
      const select = $(`[data-task-version="${taskId}"]`);
      if (!select) return;

      select.innerHTML = task.variants
        .map((variant) => `<option value="${escapeHtml(variant.id)}" ${variant.id === task.default ? 'selected' : ''}>${escapeHtml(variant.label)}</option>`)
        .join('');
    });

    resumeActiveJob(data.jobs || []);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function startTask(button) {
  if (currentJobId) {
    showToast(
      '이미 작업이 실행 중입니다.',
      true,
    );
    return;
  }

  const taskElement = button.closest('.task');
  const logElement = $('pre', taskElement);
  setButtonBusy(button, true, 'CUDA 확인 중…');
  $('em', taskElement).textContent = '환경 확인 중';
  logElement.textContent = '$ CUDA 환경과 실행 가능 여부를 확인하는 중…';

  try {
    const job = await api(
      `/api/tasks/${button.dataset.task}`,
      {
        method: 'POST',
        body: JSON.stringify({
          variant: $(`[data-task-version="${button.dataset.task}"]`)?.value,
        }),
      },
    );

    currentJobId = job.id;

    setButtonBusy(button, true, '실행 중…');

    $('em', taskElement).textContent =
      '실행 중';

    $('pre', taskElement).textContent =
      '$ 준비 중…';

    $(`[data-stop-task="${button.dataset.task}"]`).disabled = false;

    pollJob(taskElement);
  } catch (error) {
    setButtonBusy(button, false);

    $('em', taskElement).textContent = '실패';
    logElement.textContent = [
      `$ 오류: ${error.message}`,
      ...(error.data?.log || []),
    ].join('\n');
    logElement.scrollTop = logElement.scrollHeight;
    showToast(error.message, true);
  }
}


function resumeActiveJob(jobs) {
  if (currentJobId) return;

  const activeJob = [...jobs].reverse().find((job) =>
    ['queued', 'running', 'stopping'].includes(job.status),
  );

  if (!activeJob) return;

  currentJobId = activeJob.id;

  if (activeJob.task === 'script') {
    setButtonBusy($('#run-script'), true, '실행 중…');
    $('#stop-script').disabled = activeJob.status === 'stopping';
    if (activeJob.status === 'stopping') {
      setButtonBusy($('#stop-script'), true, '중지 중…');
    }
    pollScriptJob();
    return;
  }

  const taskElement = $(`#${activeJob.task}`);
  if (taskElement) {
    const stopButton = $('[data-stop-task]', taskElement);
    setButtonBusy($('[data-task]', taskElement), true, '실행 중…');
    stopButton.disabled = activeJob.status === 'stopping';
    if (activeJob.status === 'stopping') {
      setButtonBusy(stopButton, true, '중지 중…');
    }
    pollJob(taskElement);
  }
}


async function loadScripts() {
  try {
    const data = await api('/api/scripts');
    $('#script-select').innerHTML = data.scripts
      .map((script) => `<option value="${escapeHtml(script.path)}">[${escapeHtml(script.group)}] ${escapeHtml(script.name)}</option>`)
      .join('');
    $('#script-status').textContent = `${data.scripts.length}개 파일 탐색됨`;
  } catch (error) {
    $('#script-status').textContent = error.message;
  }
}


async function runSelectedScript() {
  if (currentJobId) {
    showToast('이미 작업이 실행 중입니다.', true);
    return;
  }

  const runButton = $('#run-script');
  setButtonBusy(runButton, true, '시작 중…');
  $('#script-job-status').textContent = '시작 중';
  $('#script-log').textContent = '$ 스크립트 실행을 준비하는 중…';

  try {
    const job = await api('/api/scripts/run', {
      method: 'POST',
      body: JSON.stringify({ path: $('#script-select').value }),
    });
    currentJobId = job.id;
    setButtonBusy(runButton, true, '실행 중…');
    $('#script-job-status').textContent = '실행 중';
    $('#script-log').textContent = '$ 준비 중…';
    $('#stop-script').disabled = false;
    pollScriptJob();
  } catch (error) {
    setButtonBusy(runButton, false);
    $('#script-job-status').textContent = '실패';
    $('#script-log').textContent = [
      `$ 오류: ${error.message}`,
      ...(error.data?.log || []),
    ].join('\n');
    showToast(error.message, true);
  }
}


async function pollScriptJob() {
  clearTimeout(jobPollTimer);
  try {
    const job = await api(`/api/jobs/${currentJobId}`);
    $('#script-job-status').textContent = JOB_STATUS_TEXT[job.status] || job.status;
    $('#script-log').textContent = (job.log || []).join('\n') || '$ 프로세스 시작 중…';
    if (['success', 'failed', 'cancelled'].includes(job.status)) {
      const failed = job.status === 'failed';
      showToast(job.status === 'cancelled' ? '스크립트 실행 중지됨' : job.status === 'success' ? '스크립트 실행 완료' : '스크립트 실행 실패', failed);
      setButtonBusy($('#run-script'), false);
      setButtonBusy($('#stop-script'), false);
      $('#stop-script').disabled = true;
      currentJobId = null;
      return;
    }
    jobPollTimer = setTimeout(pollScriptJob, 1500);
  } catch (error) {
    $('#script-job-status').textContent = '실패';
    $('#script-log').textContent += `\n$ 로그 조회 오류: ${error.message}`;
    setButtonBusy($('#run-script'), false);
    setButtonBusy($('#stop-script'), false);
    $('#stop-script').disabled = true;
    showToast(error.message, true);
    currentJobId = null;
  }
}


async function loadStorage() {
  try {
    const storage = await api('/api/storage');
    $('#storage-status').textContent = storage.configured
      ? `연결됨: ${storage.external}`
      : '미연결 — ARTIFACT_STORAGE_PATH를 설정하세요.';
    $('#storage-pull').disabled = !storage.configured;
    $('#storage-push').disabled = !storage.configured;
  } catch (error) {
    $('#storage-status').textContent = error.message;
  }
}


async function syncStorage(direction) {
  try {
    const result = await api('/api/storage/sync', {
      method: 'POST',
      body: JSON.stringify({ direction }),
    });
    showToast(`${result.files}개 파일을 동기화했습니다.`);
  } catch (error) {
    showToast(error.message, true);
  }
}


async function pollJob(taskElement) {
  clearTimeout(jobPollTimer);

  try {
    const job = await api(
      `/api/jobs/${currentJobId}`,
    );

    const logElement =
      $('pre', taskElement);

    $('em', taskElement).textContent =
      JOB_STATUS_TEXT[job.status] ||
      job.status;

    logElement.textContent =
      (job.log || []).join('\n') ||
      '$ 프로세스 시작 중…';

    logElement.scrollTop =
      logElement.scrollHeight;

    const isFinished = [
      'success',
      'failed',
      'cancelled',
    ].includes(job.status);

    if (isFinished) {
      const failed =
        job.status === 'failed';

      showToast(
        job.status === 'cancelled'
          ? '작업 중지됨'
          : failed
          ? '작업 실패'
          : '작업 완료',
        failed,
      );

      setButtonBusy($('[data-task]', taskElement), false);
      setButtonBusy($('[data-stop-task]', taskElement), false);
      $('[data-stop-task]', taskElement).disabled = true;
      currentJobId = null;
      return;
    }

    jobPollTimer = setTimeout(
      () => pollJob(taskElement),
      1500,
    );
  } catch (error) {
    const logElement = $('pre', taskElement);
    $('em', taskElement).textContent = '실패';
    logElement.textContent += `\n$ 로그 조회 오류: ${error.message}`;
    logElement.scrollTop = logElement.scrollHeight;
    setButtonBusy($('[data-task]', taskElement), false);
    setButtonBusy($('[data-stop-task]', taskElement), false);
    $('[data-stop-task]', taskElement).disabled = true;
    showToast(error.message, true);
    currentJobId = null;
  }
}


async function stopCurrentJob(stopButton) {
  if (!currentJobId) return;

  setButtonBusy(stopButton, true, '중지 중…');

  try {
    await api(`/api/jobs/${currentJobId}/stop`, {
      method: 'POST',
      body: '{}',
    });
    showToast('중지 요청을 보냈습니다.');
  } catch (error) {
    setButtonBusy(stopButton, false);
    showToast(error.message, true);
  }
}


function initTasks() {
  $$('[data-task]').forEach((button) => {
    button.onclick = () => {
      startTask(button);
    };
  });

  $$('[data-stop-task]').forEach((button) => {
    button.onclick = () => stopCurrentJob(button);
  });
}


// ============================================================
// Refresh
// ============================================================

function refreshAll() {
  loadGit();
  loadIssues();
  loadNotes();
  loadTasks();
  loadScripts();
  loadStorage();

  showToast('새로고침했습니다.');
}


// ============================================================
// Event Binding
// ============================================================

function bindEvents() {
  $('#pull').onclick =
    pullGit;

  $('#branches').onclick =
    openBranchModal;

  $('#close').onclick =
    () => $('#modal').close();

  $('#refresh').onclick =
    refreshAll;

  $('#run-script').onclick =
    runSelectedScript;

  $('#stop-script').onclick =
    () => stopCurrentJob($('#stop-script'));

  $('#storage-pull').onclick =
    () => syncStorage('pull');

  $('#storage-push').onclick =
    () => syncStorage('push');
}


// ============================================================
// Initialization
// ============================================================

async function initAdmin() {
  const dialog = $('#admin-dialog');
  const form = $('#admin-form');
  const login = $('#admin-login');
  const logout = $('#admin-logout');
  const errorMessage = $('#admin-error');
  const submit = $('button[type="submit"]', form);

  login.onclick = () => {
    errorMessage.hidden = window.isSecureContext;
    errorMessage.textContent = window.isSecureContext ? '' : 'HTTPS 주소에서 로그인하세요.';
    submit.disabled = !window.isSecureContext;
    dialog.showModal();
  };
  $('#admin-cancel').onclick = () => dialog.close();
  dialog.addEventListener('close', () => { $('#admin-password').value = ''; });

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    if (!window.isSecureContext || submit.disabled) return;
    setButtonBusy(submit, true, '로그인 중…');
    errorMessage.hidden = true;
    const credentials = Object.fromEntries(new FormData(form));
    $('#admin-password').value = '';
    try {
      await api('/api/auth/login', {method: 'POST', body: JSON.stringify(credentials)});
      const session = await api('/api/auth/session');
      if (!session.authenticated) throw new Error('브라우저가 로그인 쿠키를 저장하지 못했습니다.');
      history.replaceState(null, '', '#home');
      window.location.reload();
    } catch (error) {
      errorMessage.textContent = error.message;
      errorMessage.hidden = false;
    } finally {
      setButtonBusy(submit, false);
    }
  });

  logout.onclick = async () => {
    setButtonBusy(logout, true, '로그아웃 중…');
    try {
      await api('/api/auth/logout', {method: 'POST', body: '{}'});
      history.replaceState(null, '', '#production');
      window.location.reload();
    } catch (error) {
      showToast(error.message, true);
      setButtonBusy(logout, false);
    }
  };

  try {
    ({authenticated: isAdmin} = await api('/api/auth/session'));
  } catch {
    // Keep the chat available even when administrator status is unavailable.
  }
  $$('[data-admin-only]').forEach((element) => { element.hidden = !isAdmin; });
  login.hidden = isAdmin;
  logout.hidden = !isAdmin;
  if (isAdmin) {
    $('.brand').href = '#home';
    window.addEventListener('pageshow', (event) => {
      if (event.persisted) window.location.reload();
    });
  }
}

async function init() {
  initTheme();
  await initAdmin();
  $('#paper-schema-retry').onclick = loadPaperStructure;
  initNavigation();
  if (!isAdmin) return;
  initColabLinks();
  initNotes();
  initTasks();
  bindEvents();

  loadGit();
  loadIssues();
  loadNotes();
  loadTasks();
  loadScripts();
  loadStorage();
}

init();
