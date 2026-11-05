/* ==========================================================================
   DocuMind – script.js
   Frontend controller for the supplied index.html + style.css.

   IMPORTANT BACKEND CONTRACT
   --------------------------
   The API paths below are proposed integration paths, not a claim that
   the FastAPI backend already implements them. Adjust API_ROUTES when
   backend/main.py and its routes are ready.

   Expected response shapes:
   GET  /health                 -> { "status": "ok" }
   GET  /api/documents          -> { "documents": [{id, filename, size, uploaded_at, status}] }
   POST /api/documents/upload   -> multipart field "file"; returns {document:{id,...}}
   POST /api/chat               -> {answer, sources:[{page, text, title}]}
   POST /api/summarize          -> {summary}
   POST /api/quiz/generate      -> {questions:[{question, options, answer_index, explanation}]}
   POST /api/compare            -> {similarities:[], differences:[]}
   DELETE /api/documents/{id}   -> success response

   No PDF, API key, or AI result is stored in localStorage.
   ========================================================================== */

(() => {
    'use strict';

    const STORAGE_KEYS = {
        settings: 'documind.settings.v1',
        chats: 'documind.chats.v1',
        activeChat: 'documind.activeChat.v1',
        counters: 'documind.counters.v1'
    };

    const DEFAULT_SETTINGS = {
        name: 'My Account',
        theme: 'system',
        apiUrl: 'http://localhost:8000',
        chunkSize: 1000,
        topK: 4,
        showSources: true
    };

    const API_ROUTES = {
        health: '/health',
        documents: '/api/documents',
        upload: '/api/documents/upload',
        chat: '/api/chat',
        summarize: '/api/summarize',
        quiz: '/api/quiz/generate',
        compare: '/api/compare'
    };

    const PAGE_META = {
        dashboard: ['Dashboard', 'Your intelligent document workspace'],
        documents: ['My Documents', 'Manage your indexed PDF library'],
        assistant: ['AI Assistant', 'Ask questions grounded in your documents'],
        summarizer: ['Smart Summarizer', 'Turn long PDFs into concise notes'],
        quiz: ['Quiz Generator', 'Test your understanding with document-based questions'],
        compare: ['Compare PDFs', 'Find similarities and differences between documents'],
        settings: ['Settings', 'Customize your DocuMind workspace']
    };

    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
    const byId = id => document.getElementById(id);

    const state = {
        settings: loadJSON(STORAGE_KEYS.settings, DEFAULT_SETTINGS),
        documents: [],
        chats: loadJSON(STORAGE_KEYS.chats, []),
        activeChatId: localStorage.getItem(STORAGE_KEYS.activeChat) || null,
        counters: loadJSON(STORAGE_KEYS.counters, {
            questions: 0,
            summaries: 0,
            quizzes: 0
        }),
        currentQuiz: [],
        quizAnswers: {},
        quizSubmitted: false,
        documentView: 'grid',
        searchTerm: '',
        healthChecked: false,
        requestControllers: new Set()
    };

    state.settings = { ...DEFAULT_SETTINGS, ...state.settings };

    if (!Array.isArray(state.chats)) {
        state.chats = [];
    }

    if (!state.counters || typeof state.counters !== 'object') {
        state.counters = {
            questions: 0,
            summaries: 0,
            quizzes: 0
        };
    }

    function loadJSON(key, fallback) {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : { ...fallback };
        } catch (error) {
            console.warn(`DocuMind: could not read ${key}`, error);
            return Array.isArray(fallback) ? [] : { ...fallback };
        }
    }

    function saveJSON(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.warn(`DocuMind: could not save ${key}`, error);
            toast('Browser storage is unavailable or full.', 'warning');
            return false;
        }
    }

    function escapeHTML(value) {
        return String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;'
        })[char]);
    }

    function safeText(value, fallback = '') {
        return typeof value === 'string' && value.trim()
            ? value.trim()
            : fallback;
    }

    function formatBytes(bytes) {
        const size = Number(bytes);

        if (!Number.isFinite(size) || size <= 0) {
            return 'Size unavailable';
        }

        if (size < 1024 * 1024) {
            return `${(size / 1024).toFixed(1)} KB`;
        }

        return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    }

    function formatDate(value) {
        if (!value) return 'Date unavailable';

        const date = new Date(value);

        return Number.isNaN(date.getTime())
            ? 'Date unavailable'
            : new Intl.DateTimeFormat(undefined, {
                dateStyle: 'medium'
            }).format(date);
    }

    function normaliseDocument(item) {
        const source = item?.document || item || {};

        return {
            id: String(
                source.id ??
                source.document_id ??
                source._id ??
                ''
            ),
            name: safeText(
                source.filename ||
                source.file_name ||
                source.name,
                'Untitled PDF'
            ),
            size: Number(
                source.size ??
                source.file_size ??
                0
            ),
            uploadedAt:
                source.uploaded_at ||
                source.created_at ||
                source.upload_date ||
                null,
            status: safeText(source.status, 'Indexed'),
            pages: source.pages ?? source.page_count ?? null
        };
    }

    function apiURL(path) {
        const base = String(
            state.settings.apiUrl || DEFAULT_SETTINGS.apiUrl
        ).trim().replace(/\/+$/, '');

        return `${base}${path.startsWith('/') ? path : `/${path}`}`;
    }

    async function apiRequest(path, options = {}) {
        const controller = new AbortController();

        state.requestControllers.add(controller);

        const timeout = window.setTimeout(
            () => controller.abort(),
            options.timeout || 45000
        );

        try {
            const headers = new Headers(options.headers || {});

            if (
                options.body &&
                !(options.body instanceof FormData) &&
                !headers.has('Content-Type')
            ) {
                headers.set('Content-Type', 'application/json');
            }

            const response = await fetch(apiURL(path), {
                ...options,
                headers,
                signal: options.signal || controller.signal
            });

            const contentType =
                response.headers.get('content-type') || '';

            const payload = contentType.includes('application/json')
                ? await response.json()
                : await response.text();

            if (!response.ok) {
                const detail = typeof payload === 'string'
                    ? payload
                    : payload?.detail ||
                      payload?.message ||
                      payload?.error;

                throw new Error(
                    detail || `Request failed (${response.status})`
                );
            }

            return payload;

        } finally {
            window.clearTimeout(timeout);
            state.requestControllers.delete(controller);
        }
    }

    function toast(message, type = 'info', duration = 3600) {
        const container = byId('toastContainer');

        if (!container) return;

        const item = document.createElement('div');

        item.className = `toast toast-${type}`;
        item.setAttribute(
            'role',
            type === 'error' ? 'alert' : 'status'
        );
        item.textContent = message;

        container.appendChild(item);

        window.setTimeout(() => {
            item.classList.add('is-leaving');

            window.setTimeout(() => item.remove(), 220);
        }, duration);
    }

    function setButtonLoading(button, loading, label) {
        if (!button) return;

        button.classList.toggle('loading', loading);
        button.disabled = loading;

        if (label) {
            button.setAttribute(
                'aria-label',
                loading
                    ? label
                    : button.dataset.originalAriaLabel || label
            );
        }
    }

    function refreshIcons() {
        // Lucide is loaded with defer; wait until it exists.
        if (
            window.lucide &&
            typeof window.lucide.createIcons === 'function'
        ) {
            window.lucide.createIcons();
        }
    }

    /* ----------------------------- Navigation ---------------------------- */

    function navigate(page, updateHash = true) {
        if (!PAGE_META[page]) {
            page = 'dashboard';
        }

        $$('[data-page-id]').forEach(section => {
            const active = section.dataset.pageId === page;

            section.classList.toggle('active', active);
            section.hidden = !active;
        });

        $$('[data-page]').forEach(link => {
            const active = link.dataset.page === page;

            link.classList.toggle('active', active);

            if (active) {
                link.setAttribute('aria-current', 'page');
            } else {
                link.removeAttribute('aria-current');
            }
        });

        const [title, subtitle] = PAGE_META[page];

        byId('pageTitle').textContent = title;
        byId('pageSubtitle').textContent = subtitle;

        if (
            updateHash &&
            location.hash !== `#${page}`
        ) {
            history.replaceState(null, '', `#${page}`);
        }

        closeSidebar();

        if (page === 'assistant') {
            byId('chatInput')?.focus({ preventScroll: true });
        }

        if (page === 'documents') {
            renderDocuments();
        }

        if (page === 'assistant') {
            renderActiveChat();
        }

        refreshIcons();
    }

    function handleRoute() {
        const page = location.hash.replace('#', '');

        navigate(
            PAGE_META[page] ? page : 'dashboard',
            false
        );
    }


    function openFeature(feature) {
        const pages = {
            chat: 'assistant',
            summary: 'summarizer',
            quiz: 'quiz',
            compare: 'compare'
        };

        navigate(pages[feature] || 'assistant');

        if (feature !== 'chat' && state.documents.length === 0) {
            toast(
                'Upload and index a PDF before using this feature.',
                'info'
            );
        }
    }

    /* ------------------------------- Theme -------------------------------- */

    function applyTheme(theme) {
        const effective = theme === 'system'
            ? (
                window.matchMedia?.('(prefers-color-scheme: dark)').matches
                    ? 'dark'
                    : 'light'
            )
            : theme;

        document.documentElement.dataset.theme = effective;

        const toggle = byId('themeToggle');

        if (toggle) {
            toggle.setAttribute(
                'aria-pressed',
                String(effective === 'dark')
            );

            toggle.setAttribute(
                'aria-label',
                `Switch to ${effective === 'dark' ? 'light' : 'dark'} theme`
            );
        }

        const select = byId('settingsTheme');

        if (select && select.value !== theme) {
            select.value = theme;
        }

        refreshIcons();
    }

    function setTheme(theme) {
        state.settings.theme = ['light', 'dark', 'system'].includes(theme)
            ? theme
            : 'system';

        saveJSON(STORAGE_KEYS.settings, state.settings);
        applyTheme(state.settings.theme);
    }

    /* ---------------------------- Sidebar/mobile -------------------------- */

    function openSidebar() {
        byId('sidebar')?.classList.add('open');
        byId('sidebarOverlay')?.removeAttribute('hidden');

        document.body.classList.add('sidebar-open');

        byId('sidebarToggle')?.setAttribute(
            'aria-expanded',
            'true'
        );
    }

    function closeSidebar() {
        byId('sidebar')?.classList.remove('open');
        byId('sidebarOverlay')?.setAttribute('hidden', '');

        document.body.classList.remove('sidebar-open');

        byId('sidebarToggle')?.setAttribute(
            'aria-expanded',
            'false'
        );
    }

    /* ------------------------------- Upload -------------------------------- */

    function setFileStatus(message, type = '') {
        const el = byId('fileStatus');

        if (!el) return;

        el.textContent = message;
        el.classList.remove('success', 'error');

        if (type) {
            el.classList.add(type);
        }
    }

    function setUploadProgress(percent, filename) {
        const wrapper = byId('uploadProgress');
        const fill = byId('uploadProgressFill');
        const label = byId('uploadPercent');
        const bar = byId('uploadProgressBar');

        wrapper.hidden = false;
        byId('uploadFileName').textContent = filename;

        label.textContent = `${Math.round(percent)}%`;

        fill.style.width =
            `${Math.max(0, Math.min(100, percent))}%`;

        bar.setAttribute(
            'aria-valuenow',
            String(Math.round(percent))
        );
    }

    function resetUploadProgress() {
        byId('uploadProgress').hidden = true;
        byId('uploadProgressFill').style.width = '0%';

        byId('uploadProgressBar').setAttribute(
            'aria-valuenow',
            '0'
        );
    }

    function validatePDF(file) {
        if (!file) {
            return 'Please choose a PDF file.';
        }

        const isPDF =
            file.type === 'application/pdf' ||
            file.name.toLowerCase().endsWith('.pdf');

        if (!isPDF) {
            return 'Only PDF files are supported.';
        }

        if (file.size > 20 * 1024 * 1024) {
            return 'The PDF is larger than the 20 MB limit.';
        }

        if (file.size === 0) {
            return 'This file is empty.';
        }

        return '';
    }

    function uploadPDF(file) {
        const problem = validatePDF(file);

        if (problem) {
            setFileStatus(problem, 'error');
            toast(problem, 'error');
            byId('pdfInput').value = '';
            return;
        }

        const zone = byId('dropZone');

        zone.classList.remove('error', 'success');

        setFileStatus('Connecting to the DocuMind backend…');
        setUploadProgress(0, file.name);

        // XMLHttpRequest provides upload progress.
        // The backend must accept multipart/form-data
        // using the field name "file".

        const xhr = new XMLHttpRequest();

        xhr.open(
            'POST',
            apiURL(API_ROUTES.upload),
            true
        );

        xhr.responseType = 'text';
        xhr.timeout = 120000;

        xhr.upload.addEventListener('progress', event => {
            if (event.lengthComputable) {
                setUploadProgress(
                    (event.loaded / event.total) * 100,
                    file.name
                );
            }
        });

        xhr.addEventListener('load', async () => {
            let payload = {};

            try {
                payload = xhr.responseText
                    ? JSON.parse(xhr.responseText)
                    : {};
            } catch (_) {
                payload = { detail: xhr.responseText };
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                const message =
                    payload.detail ||
                    payload.message ||
                    `Upload failed (${xhr.status}).`;

                zone.classList.add('error');
                setFileStatus(message, 'error');
                toast(message, 'error');

                resetUploadProgress();
                byId('pdfInput').value = '';

                return;
            }

            const doc = normaliseDocument(
                payload.document || payload
            );

            if (!doc.id) {
                setFileStatus(
                    'The server accepted the file but returned no document ID. Check the backend response.',
                    'error'
                );

                toast(
                    'Unexpected upload response: document ID is missing.',
                    'error'
                );
            } else {
                setUploadProgress(100, file.name);

                zone.classList.add('success');

                setFileStatus(
                    `${doc.name} uploaded and processed successfully.`,
                    'success'
                );

                toast('PDF uploaded successfully.', 'success');

                await loadDocuments(false);
                navigate('documents');
            }

            window.setTimeout(resetUploadProgress, 1400);
            byId('pdfInput').value = '';
        });

        xhr.addEventListener('error', () => {
            zone.classList.add('error');

            setFileStatus(
                'Could not reach the backend. Your PDF has not been uploaded.',
                'error'
            );

            toast(
                'Backend unavailable. Start FastAPI and try again.',
                'error'
            );

            resetUploadProgress();
            byId('pdfInput').value = '';
        });

        xhr.addEventListener('timeout', () => {
            setFileStatus(
                'Upload timed out. Check the backend and try again.',
                'error'
            );

            toast('Upload timed out.', 'error');

            resetUploadProgress();
            byId('pdfInput').value = '';
        });

        const formData = new FormData();

        formData.append('file', file, file.name);

        xhr.send(formData);
    }

    /* ---------------------------- Documents UI ---------------------------- */

    function emptyState(title, description, actionHTML = '') {
        return `
            <div class="empty-state">
                <div class="empty-icon">
                    <i data-lucide="file-search"></i>
                </div>
                <h4>${escapeHTML(title)}</h4>
                <p>${escapeHTML(description)}</p>
                ${actionHTML}
            </div>
        `;
    }

    function documentMarkup(doc, compact = false) {
        const id = escapeHTML(doc.id);
        const name = escapeHTML(doc.name);

        const meta = [
            formatBytes(doc.size),
            formatDate(doc.uploadedAt),
            safeText(doc.status, 'Indexed')
        ].join(' · ');

        return `
            <article class="document-item" data-document-id="${id}">
                <div class="document-icon" aria-hidden="true">
                    <i data-lucide="file-pdf"></i>
                </div>

                <div class="document-info">
                    <strong class="document-name" title="${name}">
                        ${name}
                    </strong>

                    <small class="document-meta">
                        ${escapeHTML(meta)}
                    </small>
                </div>

                <div class="document-actions">
                    <button
                        class="icon-btn"
                        type="button"
                        data-action="ask"
                        data-id="${id}"
                        aria-label="Ask questions about ${name}"
                        title="Ask AI"
                    >
                        <i data-lucide="message-square-text"></i>
                    </button>

                    ${
                        compact
                            ? ''
                            : `
                                <button
                                    class="icon-btn"
                                    type="button"
                                    data-action="delete"
                                    data-id="${id}"
                                    aria-label="Delete ${name}"
                                    title="Delete document"
                                >
                                    <i data-lucide="trash-2"></i>
                                </button>
                            `
                    }
                </div>
            </article>
        `;
    }



    function renderDocuments() {
        const recent = byId('documentList');
        const all = byId('allDocumentsList');

        const filtered = state.documents.filter(doc =>
            doc.name.toLowerCase().includes(state.searchTerm)
        );

        const sorted = [...filtered].sort((a, b) => {
            const sort = byId('documentsSort')?.value || 'recent';

            if (sort === 'name') {
                return a.name.localeCompare(b.name);
            }

            if (sort === 'size') {
                return b.size - a.size;
            }

            return new Date(b.uploadedAt || 0) -
                new Date(a.uploadedAt || 0);
        });

        recent.innerHTML = state.documents.length
            ? [...state.documents]
                .slice(0, 5)
                .map(doc => documentMarkup(doc, true))
                .join('')
            : emptyState(
                'No documents yet',
                'Your uploaded documents will appear here.'
            );

        all.classList.toggle(
            'document-grid',
            state.documentView === 'grid'
        );

        all.classList.toggle(
            'list-view',
            state.documentView === 'list'
        );

        all.innerHTML = sorted.length
            ? sorted.map(doc => documentMarkup(doc)).join('')
            : emptyState(
                state.documents.length
                    ? 'No matching documents'
                    : 'Your library is empty',
                state.documents.length
                    ? 'Try another search term.'
                    : 'Upload a PDF to start asking questions.',
                state.documents.length
                    ? ''
                    : `
                        <button class="btn btn-secondary"
                                type="button"
                                data-action="browse">
                            <i data-lucide="upload"></i>
                            <span>Upload your first PDF</span>
                        </button>
                    `
            );

        byId('statDocuments').textContent =
            String(state.documents.length);

        byId('navDocumentCount').textContent =
            String(state.documents.length);

        populateDocumentSelects();
        refreshIcons();
    }

    function populateDocumentSelects() {
        const ids = [
            'chatDocumentSelect',
            'summaryDocumentSelect',
            'quizDocumentSelect',
            'compareDocumentA',
            'compareDocumentB'
        ];

        ids.forEach(id => {
            const select = byId(id);

            if (!select) return;

            const previous = select.value;

            const first = select.options[0]?.outerHTML ||
                '<option value="">Select a document…</option>';

            select.innerHTML = first +
                state.documents.map(doc => `
                    <option value="${escapeHTML(doc.id)}">
                        ${escapeHTML(doc.name)}
                    </option>
                `).join('');

            if (state.documents.some(doc => doc.id === previous)) {
                select.value = previous;
            }
        });
    }

    async function loadDocuments(showMessage = true) {
        try {
            const result = await apiRequest(
                API_ROUTES.documents,
                {
                    method: 'GET',
                    timeout: 15000
                }
            );

            const list = Array.isArray(result)
                ? result
                : result.documents || result.items || [];

            state.documents = list
                .map(normaliseDocument)
                .filter(doc => doc.id);

            renderDocuments();

            if (showMessage) {
                toast(
                    `${state.documents.length} document(s) loaded.`,
                    'success'
                );
            }

        } catch (error) {
            state.documents = [];
            renderDocuments();

            if (showMessage) {
                toast(
                    `Could not load documents: ${error.message}`,
                    'error'
                );
            }
        }
    }

    async function deleteDocument(id) {
        const doc = state.documents.find(item => item.id === id);

        if (!doc) return;

        if (!window.confirm(
            `Delete "${doc.name}" from your DocuMind library? This cannot be undone.`
        )) {
            return;
        }

        try {
            await apiRequest(
                `${API_ROUTES.documents}/${encodeURIComponent(id)}`,
                { method: 'DELETE' }
            );

            state.documents = state.documents.filter(
                item => item.id !== id
            );

            renderDocuments();
            toast('Document deleted.', 'success');

        } catch (error) {
            toast(
                `Could not delete document: ${error.message}`,
                'error'
            );
        }
    }

    /* -------------------------------- Chats -------------------------------- */

    function makeId() {
        return window.crypto?.randomUUID
            ? window.crypto.randomUUID()
            : `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }

    function createChat() {
        const chat = {
            id: makeId(),
            title: 'New conversation',
            documentId: '',
            messages: [],
            updatedAt: new Date().toISOString()
        };

        state.chats.unshift(chat);
        state.activeChatId = chat.id;

        persistChats();
        renderChatHistory();
        renderActiveChat();
        navigate('assistant');

        byId('chatInput')?.focus();
    }

    function activeChat() {
        return state.chats.find(
            chat => chat.id === state.activeChatId
        ) || null;
    }

    function persistChats() {
        saveJSON(STORAGE_KEYS.chats, state.chats);

        if (state.activeChatId) {
            localStorage.setItem(
                STORAGE_KEYS.activeChat,
                state.activeChatId
            );
        } else {
            localStorage.removeItem(STORAGE_KEYS.activeChat);
        }
    }

    function renderChatHistory() {
        const list = byId('chatHistory');

        if (!state.chats.length) {
            list.innerHTML = `
                <li class="chat-history-empty" id="chatHistoryEmpty">
                    No conversations yet.
                </li>
            `;
            return;
        }

        list.innerHTML = state.chats.slice(0, 30).map(chat => `
            <li>
                <button
                    type="button"
                    class="chat-history-item ${
                        chat.id === state.activeChatId ? 'active' : ''
                    }"
                    data-action="select-chat"
                    data-id="${escapeHTML(chat.id)}"
                    title="${escapeHTML(chat.title)}"
                >
                    <i data-lucide="message-square"></i>
                    <span>${escapeHTML(chat.title)}</span>
                </button>
            </li>
        `).join('');

        refreshIcons();
    }

    function renderMessage(message) {
        const isUser = message.role === 'user';

        const wrapper = document.createElement('div');

        wrapper.className =
            `message ${isUser ? 'user' : 'assistant'}` +
            `${message.error ? ' error' : ''}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';

        avatar.textContent = isUser
            ? (state.settings.name?.[0] || 'Y').toUpperCase()
            : 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';

        const paragraph = document.createElement('p');
        paragraph.textContent = message.content || '';

        content.appendChild(paragraph);

        if (
            Array.isArray(message.sources) &&
            message.sources.length &&
            state.settings.showSources
        ) {
            const sourceLine = document.createElement('div');

            sourceLine.className = 'message-sources';

            sourceLine.textContent = message.sources.map(source =>
                `Page ${source.page ?? source.page_number ?? '?'}`
            ).join(' · ');

            content.appendChild(sourceLine);
        }

        wrapper.append(avatar, content);

        return wrapper;
    }

    function renderActiveChat() {
        const chat = activeChat();
        const messages = byId('chatMessages');

        if (!chat || !chat.messages.length) {
            messages.innerHTML = `
                <div class="empty-state" id="chatEmptyState">
                    <div class="empty-icon">
                        <i data-lucide="message-square-text"></i>
                    </div>
                    <h4>Start a conversation</h4>
                    <p>
                        Select a document and ask your first question.
                        Answers will cite the source pages.
                    </p>
                </div>
            `;
        } else {
            messages.replaceChildren(
                ...chat.messages.map(renderMessage)
            );

            messages.scrollTop = messages.scrollHeight;
        }

        if (
            chat?.documentId &&
            state.documents.some(doc => doc.id === chat.documentId)
        ) {
            byId('chatDocumentSelect').value = chat.documentId;
        }

        renderChatHistory();
        renderSources([]);
        refreshIcons();
    }

    function renderSources(sources) {
        const list = byId('sourcesList');

        if (
            !state.settings.showSources ||
            !Array.isArray(sources) ||
            !sources.length
        ) {
            list.innerHTML = `
                <div class="empty-state compact">
                    <div class="empty-icon">
                        <i data-lucide="quote"></i>
                    </div>
                    <p>
                        Retrieved passages will appear here after you ask a question.
                    </p>
                </div>
            `;

            refreshIcons();
            return;
        }

        list.innerHTML = sources.map((source, index) => {
            const page =
                source.page ??
                source.page_number ??
                source.metadata?.page ??
                '?';

            const title =
                source.title ||
                source.filename ||
                `Source ${index + 1}`;

            const text =
                source.text ||
                source.content ||
                source.chunk ||
                '';

            return `
                <article class="source-card">
                    <div class="source-header">
                        <strong class="source-title">
                            ${escapeHTML(title)}
                        </strong>

                        <span class="source-page">
                            Page ${escapeHTML(page)}
                        </span>
                    </div>

                    <p class="source-text">
                        ${escapeHTML(text)}
                    </p>
                </article>
            `;
        }).join('');
    }

    function showTyping() {
        const typing = document.createElement('div');

        typing.className = 'message assistant';
        typing.id = 'typingMessage';

        typing.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;

        byId('chatMessages').appendChild(typing);

        byId('chatMessages').scrollTop =
            byId('chatMessages').scrollHeight;
    }



    async function sendQuestion(event) {
        event.preventDefault();

        const input = byId('chatInput');
        const question = input.value.trim();
        const documentId = byId('chatDocumentSelect').value;

        if (!question) return;

        if (!documentId) {
            toast(
                'Select an indexed PDF before asking a question.',
                'warning'
            );

            byId('chatDocumentSelect').focus();
            return;
        }

        let chat = activeChat();

        if (!chat) {
            chat = {
                id: makeId(),
                title: question.slice(0, 42),
                documentId,
                messages: [],
                updatedAt: new Date().toISOString()
            };

            state.chats.unshift(chat);
            state.activeChatId = chat.id;
        }

        chat.documentId = documentId;

        if (!chat.messages.length) {
            chat.title = question.slice(0, 42);
        }

        chat.messages.push({
            role: 'user',
            content: question
        });

        chat.updatedAt = new Date().toISOString();

        input.value = '';
        input.style.height = 'auto';

        renderActiveChat();
        showTyping();

        setButtonLoading(
            byId('chatSendBtn'),
            true,
            'Sending question'
        );

        try {
            const result = await apiRequest(
                API_ROUTES.chat,
                {
                    method: 'POST',
                    body: JSON.stringify({
                        document_id: documentId,
                        question,
                        top_k: Number(state.settings.topK) || 4,
                        show_sources: Boolean(state.settings.showSources)
                    })
                }
            );

            const answer = safeText(
                result.answer ||
                result.response ||
                result.message,
                ''
            );

            if (!answer) {
                throw new Error(
                    'The backend response did not contain an answer.'
                );
            }

            const sources = Array.isArray(result.sources)
                ? result.sources
                : (result.citations || []);

            chat.messages.push({
                role: 'assistant',
                content: answer,
                sources
            });

            renderSources(sources);

            state.counters.questions += 1;

            saveJSON(
                STORAGE_KEYS.counters,
                state.counters
            );

            updateStats();

        } catch (error) {
            chat.messages.push({
                role: 'assistant',
                content: `I couldn't get an answer from the backend. ${error.message}`,
                error: true
            });

            renderSources([]);

        } finally {
            byId('typingMessage')?.remove();

            setButtonLoading(
                byId('chatSendBtn'),
                false,
                'Send question'
            );

            persistChats();
            renderActiveChat();

            renderSources(
                chat.messages.at(-1)?.sources || []
            );
        }
    }

    /* -------------------------- Summary generation ------------------------ */

    function renderOutputMessage(container, title, message) {
        container.innerHTML = '';

        const box = document.createElement('div');
        box.className = 'output-message';

        const heading = document.createElement('h4');
        heading.textContent = title;

        const body = document.createElement('p');
        body.textContent = message;

        box.append(heading, body);
        container.appendChild(box);
    }

    let latestSummary = '';

    async function generateSummary(event) {
        event.preventDefault();

        const documentId = byId('summaryDocumentSelect').value;

        if (!documentId) {
            return toast('Select a document first.', 'warning');
        }

        const button = byId('generateSummaryBtn');

        setButtonLoading(
            button,
            true,
            'Generating summary'
        );

        renderOutputMessage(
            byId('summaryOutput'),
            'Working on your summary',
            'The backend is processing your document…'
        );

        try {
            const result = await apiRequest(
                API_ROUTES.summarize,
                {
                    method: 'POST',
                    body: JSON.stringify({
                        document_id: documentId,
                        length: byId('summaryLength').value,
                        format: byId('summaryFormat').value
                    })
                }
            );

            latestSummary = safeText(
                result.summary ||
                result.text ||
                result.result,
                ''
            );

            if (!latestSummary) {
                throw new Error(
                    'The backend response did not contain summary text.'
                );
            }

            const output = byId('summaryOutput');

            output.replaceChildren();

            const content = document.createElement('div');

            content.className = 'summary-content';
            content.textContent = latestSummary;

            output.appendChild(content);

            byId('copySummaryBtn').disabled = false;
            byId('downloadSummaryBtn').disabled = false;

            state.counters.summaries += 1;

            saveJSON(
                STORAGE_KEYS.counters,
                state.counters
            );

            updateStats();

            toast('Summary generated.', 'success');

        } catch (error) {
            latestSummary = '';

            byId('copySummaryBtn').disabled = true;
            byId('downloadSummaryBtn').disabled = true;

            renderOutputMessage(
                byId('summaryOutput'),
                'Could not generate summary',
                error.message
            );

            toast(
                `Summary failed: ${error.message}`,
                'error'
            );

        } finally {
            setButtonLoading(
                button,
                false,
                'Generate summary'
            );
        }
    }

    async function copyText(text) {
        try {
            await navigator.clipboard.writeText(text);

            toast('Copied to clipboard.', 'success');

        } catch (_) {
            const area = document.createElement('textarea');

            area.value = text;

            document.body.appendChild(area);

            area.select();

            const copied = document.execCommand('copy');

            area.remove();

            toast(
                copied
                    ? 'Copied to clipboard.'
                    : 'Could not copy text.',
                copied ? 'success' : 'error'
            );
        }
    }

    function downloadText(filename, text) {
        const blob = new Blob(
            [text],
            { type: 'text/plain;charset=utf-8' }
        );

        const url = URL.createObjectURL(blob);

        const anchor = document.createElement('a');

        anchor.href = url;
        anchor.download = filename;

        document.body.appendChild(anchor);

        anchor.click();
        anchor.remove();

        URL.revokeObjectURL(url);
    }

    /* ----------------------------- Quiz flow ------------------------------- */

    function resetQuiz() {
        state.quizAnswers = {};
        state.quizSubmitted = false;

        $$('.quiz-option').forEach(option => {
            option.classList.remove(
                'selected',
                'correct',
                'incorrect'
            );

            option.disabled = false;
        });

        $$('.quiz-feedback, .quiz-explanation').forEach(el => {
            el.remove();
        });

        byId('quizScore').hidden = true;
        byId('resetQuizBtn').disabled = true;
    }

    function renderQuiz(questions) {
        state.currentQuiz = questions.map((q, index) => ({
            question: safeText(
                q.question || q.text,
                `Question ${index + 1}`
            ),

            options: Array.isArray(q.options)
                ? q.options
                : [],

            answerIndex: Number(
                q.answer_index ??
                q.correct_index ??
                q.correctAnswer
            ),

            explanation: safeText(q.explanation, '')
        })).filter(q =>
            q.options.length >= 2 &&
            Number.isInteger(q.answerIndex) &&
            q.answerIndex >= 0 &&
            q.answerIndex < q.options.length
        );

        state.quizAnswers = {};
        state.quizSubmitted = false;

        if (!state.currentQuiz.length) {
            throw new Error(
                'The backend returned no valid multiple-choice questions.'
            );
        }

        const output = byId('quizOutput');

        output.innerHTML = state.currentQuiz.map((q, index) => `
            <section
                class="quiz-question"
                data-question-index="${index}"
            >
                <h4 class="quiz-question-title">
                    ${index + 1}. ${escapeHTML(q.question)}
                </h4>

                <div class="quiz-options">
                    ${q.options.map((option, optionIndex) => `
                        <button
                            class="quiz-option"
                            type="button"
                            data-action="quiz-option"
                            data-question="${index}"
                            data-option="${optionIndex}"
                        >
                            <span class="quiz-option-letter">
                                ${String.fromCharCode(65 + optionIndex)}
                            </span>

                            <span class="quiz-option-text">
                                ${escapeHTML(option)}
                            </span>
                        </button>
                    `).join('')}
                </div>
            </section>
        `).join('') + `
            <div class="form-actions">
                <button
                    class="btn btn-primary"
                    type="button"
                    id="submitQuizBtn"
                >
                    Submit answers
                </button>
            </div>
        `;

        byId('resetQuizBtn').disabled = false;
        byId('quizScore').hidden = true;

        refreshIcons();
    }

    function chooseQuizOption(button) {
        if (state.quizSubmitted) return;

        const questionIndex = Number(button.dataset.question);
        const optionIndex = Number(button.dataset.option);

        state.quizAnswers[questionIndex] = optionIndex;

        const section = button.closest('.quiz-question');

        $$('.quiz-option', section).forEach(option => {
            option.classList.toggle(
                'selected',
                option === button
            );
        });
    }

    function submitQuiz() {
        if (state.quizSubmitted) return;

        if (
            Object.keys(state.quizAnswers).length !==
            state.currentQuiz.length
        ) {
            toast(
                'Please answer every question before submitting.',
                'warning'
            );

            return;
        }

        state.quizSubmitted = true;

        let score = 0;

        state.currentQuiz.forEach((question, index) => {
            const selected = state.quizAnswers[index];

            const section = $(
                `[data-question-index="${index}"]`,
                byId('quizOutput')
            );

            $$('.quiz-option', section).forEach(
                (option, optionIndex) => {
                    option.disabled = true;

                    if (optionIndex === question.answerIndex) {
                        option.classList.add('correct');
                    } else if (optionIndex === selected) {
                        option.classList.add('incorrect');
                    }
                }
            );

            const correct = selected === question.answerIndex;

            if (correct) {
                score += 1;
            }

            const feedback = document.createElement('p');

            feedback.className =
                `quiz-feedback ${correct ? 'correct' : 'incorrect'}`;

            feedback.textContent = correct
                ? 'Correct answer.'
                : `Incorrect. Correct answer: ${question.options[question.answerIndex]}`;

            section.appendChild(feedback);

            if (question.explanation) {
                const explanation = document.createElement('p');

                explanation.className = 'quiz-explanation';
                explanation.textContent = question.explanation;

                section.appendChild(explanation);
            }
        });

        byId('quizScore').textContent =
            `Score: ${score} / ${state.currentQuiz.length}`;

        byId('quizScore').hidden = false;
        byId('resetQuizBtn').disabled = false;
    }



    async function generateQuiz(event) {
        event.preventDefault();

        const documentId = byId('quizDocumentSelect').value;

        if (!documentId) {
            return toast('Select a document first.', 'warning');
        }

        const button = byId('generateQuizBtn');

        setButtonLoading(
            button,
            true,
            'Generating quiz'
        );

        renderOutputMessage(
            byId('quizOutput'),
            'Generating your quiz',
            'The backend is creating questions from the selected PDF…'
        );

        try {
            const result = await apiRequest(
                API_ROUTES.quiz,
                {
                    method: 'POST',
                    body: JSON.stringify({
                        document_id: documentId,
                        count: Number(byId('quizCount').value),
                        difficulty: byId('quizDifficulty').value
                    })
                }
            );

            renderQuiz(
                result.questions || result.quiz || []
            );

            state.counters.quizzes += 1;

            saveJSON(
                STORAGE_KEYS.counters,
                state.counters
            );

            updateStats();

            toast('Quiz generated. Good luck!', 'success');

        } catch (error) {
            renderOutputMessage(
                byId('quizOutput'),
                'Could not generate quiz',
                error.message
            );

            toast(
                `Quiz failed: ${error.message}`,
                'error'
            );

        } finally {
            setButtonLoading(
                button,
                false,
                'Generate quiz'
            );
        }
    }

    /* ----------------------------- Compare PDFs ---------------------------- */

    function renderCompare(result) {
        const similarities =
            result.similarities ||
            result.common_points ||
            [];

        const differences =
            result.differences ||
            result.distinctions ||
            [];

        const normaliseList = value =>
            Array.isArray(value)
                ? value
                : (value ? [String(value)] : []);

        const makeSection = (title, className, items) => `
            <section class="compare-section ${className}">
                <h4 class="compare-section-title">
                    ${title}
                </h4>

                ${
                    items.length
                        ? `
                            <ul>
                                ${items.map(item => `
                                    <li>
                                        ${escapeHTML(
                                            typeof item === 'string'
                                                ? item
                                                : item.text ||
                                                  item.description ||
                                                  JSON.stringify(item)
                                        )}
                                    </li>
                                `).join('')}
                            </ul>
                        `
                        : '<p>No details returned by the backend.</p>'
                }
            </section>
        `;

        const output = byId('compareOutput');

        output.innerHTML = `
            <div class="compare-result">
                ${makeSection(
                    'Similarities',
                    'similarities',
                    normaliseList(similarities)
                )}

                ${makeSection(
                    'Differences',
                    'differences',
                    normaliseList(differences)
                )}
            </div>
        `;
    }

    async function compareDocuments(event) {
        event.preventDefault();

        const first = byId('compareDocumentA').value;
        const second = byId('compareDocumentB').value;

        if (!first || !second) {
            return toast(
                'Select both PDFs to compare.',
                'warning'
            );
        }

        if (first === second) {
            return toast(
                'Choose two different PDFs.',
                'warning'
            );
        }

        const button = byId('compareBtn');

        setButtonLoading(
            button,
            true,
            'Comparing documents'
        );

        renderOutputMessage(
            byId('compareOutput'),
            'Comparing documents',
            'The backend is analysing both PDFs…'
        );

        try {
            const result = await apiRequest(
                API_ROUTES.compare,
                {
                    method: 'POST',
                    body: JSON.stringify({
                        document_a_id: first,
                        document_b_id: second,
                        focus: byId('compareFocus').value.trim()
                    })
                }
            );

            renderCompare(result);

            toast('Comparison complete.', 'success');

        } catch (error) {
            renderOutputMessage(
                byId('compareOutput'),
                'Could not compare documents',
                error.message
            );

            toast(
                `Comparison failed: ${error.message}`,
                'error'
            );

        } finally {
            setButtonLoading(
                button,
                false,
                'Compare documents'
            );
        }
    }

    /* -------------------------------- Settings ----------------------------- */

    function fillSettings() {
        byId('settingsName').value =
            state.settings.name || '';

        byId('settingsTheme').value =
            state.settings.theme;

        byId('settingsApiUrl').value =
            state.settings.apiUrl;

        byId('settingsChunkSize').value =
            state.settings.chunkSize;

        byId('settingsTopK').value =
            state.settings.topK;

        byId('settingsShowSources').checked =
            Boolean(state.settings.showSources);

        byId('userName').textContent =
            state.settings.name || 'My Account';

        byId('userAvatar').textContent =
            (state.settings.name || 'A')
                .trim()[0]
                .toUpperCase();
    }

    function saveSettings(event) {
        event.preventDefault();

        const apiUrl = byId('settingsApiUrl')
            .value
            .trim()
            .replace(/\/+$/, '');

        if (!/^https?:\/\/.+/i.test(apiUrl)) {
            toast(
                'Enter a valid backend URL starting with http:// or https://.',
                'error'
            );

            byId('settingsApiUrl').focus();
            return;
        }

        const chunkSize = Number(
            byId('settingsChunkSize').value ||
            DEFAULT_SETTINGS.chunkSize
        );

        const topK = Number(
            byId('settingsTopK').value ||
            DEFAULT_SETTINGS.topK
        );

        if (
            !Number.isInteger(chunkSize) ||
            chunkSize < 100 ||
            chunkSize > 4000
        ) {
            toast(
                'Chunk size must be between 100 and 4000.',
                'error'
            );

            return;
        }

        if (
            !Number.isInteger(topK) ||
            topK < 1 ||
            topK > 20
        ) {
            toast(
                'Retrieved passages must be between 1 and 20.',
                'error'
            );

            return;
        }

        state.settings = {
            ...state.settings,

            name:
                byId('settingsName').value.trim() ||
                'My Account',

            theme: byId('settingsTheme').value,

            apiUrl,
            chunkSize,
            topK,

            showSources:
                byId('settingsShowSources').checked
        };

        saveJSON(
            STORAGE_KEYS.settings,
            state.settings
        );

        fillSettings();
        applyTheme(state.settings.theme);

        toast('Settings saved.', 'success');

        checkBackend();
    }

    async function checkBackend() {
        const status = byId('backendStatus');

        status.classList.remove('connected', 'error');
        status.classList.add('checking');

        byId('backendStatusText').textContent =
            'Checking backend…';

        try {
            const result = await apiRequest(
                API_ROUTES.health,
                {
                    method: 'GET',
                    timeout: 6000
                }
            );

            status.classList.remove('checking', 'error');
            status.classList.add('connected');

            byId('backendStatusText').textContent =
                result?.status
                    ? `Backend connected · ${result.status}`
                    : 'Backend connected';

            state.healthChecked = true;

            return true;

        } catch (_) {
            status.classList.remove('checking', 'connected');
            status.classList.add('error');

            byId('backendStatusText').textContent =
                'Backend not connected';

            state.healthChecked = true;

            return false;
        }
    }

    async function clearLocalData() {
        const confirmed = window.confirm(
            'Clear locally saved chats, counters and preferences? This does not delete PDFs from the backend.'
        );

        if (!confirmed) return;

        state.chats = [];
        state.activeChatId = null;

        state.counters = {
            questions: 0,
            summaries: 0,
            quizzes: 0
        };

        saveJSON(STORAGE_KEYS.chats, []);
        saveJSON(STORAGE_KEYS.counters, state.counters);

        localStorage.removeItem(STORAGE_KEYS.activeChat);

        // Keep API URL and theme so the user does not lose setup.
        state.settings = {
            ...DEFAULT_SETTINGS,
            apiUrl: state.settings.apiUrl,
            theme: state.settings.theme
        };

        saveJSON(
            STORAGE_KEYS.settings,
            state.settings
        );

        fillSettings();
        renderChatHistory();
        renderActiveChat();
        updateStats();

        toast(
            'Local chats and usage data cleared.',
            'success'
        );
    }

    function updateStats() {
        byId('statQuestions').textContent =
            String(state.counters.questions || 0);

        byId('statSummaries').textContent =
            String(state.counters.summaries || 0);

        byId('statQuizzes').textContent =
            String(state.counters.quizzes || 0);

        byId('statQuestionsNote').textContent =
            'Questions asked in this browser';

        byId('statSummariesNote').textContent =
            'Summaries generated in this browser';

        byId('statQuizzesNote').textContent =
            'Quizzes generated in this browser';
    }

    /* ------------------------------ Event wiring --------------------------- */

    function bindEvents() {
        // SPA-style navigation and existing inline HTML handlers.

        document.addEventListener('click', event => {
            const pageLink = event.target.closest('[data-page]');

            if (pageLink) {
                event.preventDefault();
                navigate(pageLink.dataset.page);
                return;
            }

            const action = event.target.closest('[data-action]');

            if (!action) return;

            const id = action.dataset.id;

            switch (action.dataset.action) {
                case 'browse':
                    byId('pdfInput').click();
                    break;

                case 'ask':
                    byId('chatDocumentSelect').value = id;
                    navigate('assistant');
                    break;

                case 'delete':
                    deleteDocument(id);
                    break;

                case 'select-chat':
                    state.activeChatId = id;
                    persistChats();
                    renderActiveChat();
                    navigate('assistant');
                    break;

                case 'quiz-option':
                    chooseQuizOption(action);
                    break;
            }
        });

        byId('themeToggle').addEventListener('click', () => {
            const current =
                document.documentElement.dataset.theme;

            setTheme(
                current === 'dark' ? 'light' : 'dark'
            );
        });

        byId('settingsTheme').addEventListener(
            'change',
            event => setTheme(event.target.value)
        );

        byId('sidebarToggle').addEventListener(
            'click',
            openSidebar
        );

        byId('sidebarClose').addEventListener(
            'click',
            closeSidebar
        );

        byId('sidebarOverlay').addEventListener(
            'click',
            closeSidebar
        );

        document.addEventListener('keydown', event => {
            if (event.key === 'Escape') {
                closeSidebar();
            }
        });

        byId('welcomeUploadBtn').addEventListener(
            'click',
            () => byId('pdfInput').click()
        );

        byId('documentsUploadBtn').addEventListener(
            'click',
            () => {
                navigate('dashboard');
                byId('pdfInput').click();
            }
        );

        byId('emptyUploadBtn').addEventListener(
            'click',
            () => byId('pdfInput').click()
        );

        byId('pdfInput').addEventListener(
            'change',
            event => {
                if (event.target.files?.[0]) {
                    uploadPDF(event.target.files[0]);
                }
            }
        );

        const dropZone = byId('dropZone');

        ['dragenter', 'dragover'].forEach(type => {
            dropZone.addEventListener(type, event => {
                event.preventDefault();
                dropZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(type => {
            dropZone.addEventListener(type, event => {
                event.preventDefault();
                dropZone.classList.remove('drag-over');
            });
        });

        dropZone.addEventListener('drop', event => {
            const file = event.dataTransfer?.files?.[0];

            if (file) {
                uploadPDF(file);
            }
        });




        byId('documentsSearch').addEventListener(
            'input',
            event => {
                state.searchTerm = event.target.value
                    .trim()
                    .toLowerCase();

                renderDocuments();
            }
        );

        byId('documentsSort').addEventListener(
            'change',
            renderDocuments
        );

        byId('globalSearch').addEventListener(
            'input',
            event => {
                const query = event.target.value.trim();

                if (query) {
                    byId('documentsSearch').value = query;

                    state.searchTerm = query.toLowerCase();

                    navigate('documents');
                    renderDocuments();
                }
            }
        );

        byId('viewGrid').addEventListener(
            'click',
            () => setDocumentView('grid')
        );

        byId('viewList').addEventListener(
            'click',
            () => setDocumentView('list')
        );

        byId('newChatBtn').addEventListener(
            'click',
            createChat
        );

        byId('assistantNewChatBtn').addEventListener(
            'click',
            createChat
        );

        byId('chatForm').addEventListener(
            'submit',
            sendQuestion
        );

        byId('chatDocumentSelect').addEventListener(
            'change',
            event => {
                const chat = activeChat();

                if (chat) {
                    chat.documentId = event.target.value;
                    persistChats();
                }
            }
        );

        byId('chatInput').addEventListener(
            'keydown',
            event => {
                if (
                    event.key === 'Enter' &&
                    !event.shiftKey
                ) {
                    event.preventDefault();

                    byId('chatForm').requestSubmit();
                }
            }
        );

        byId('chatInput').addEventListener(
            'input',
            event => {
                event.target.style.height = 'auto';

                event.target.style.height =
                    `${Math.min(event.target.scrollHeight, 160)}px`;
            }
        );

        byId('summaryForm').addEventListener(
            'submit',
            generateSummary
        );

        byId('copySummaryBtn').addEventListener(
            'click',
            () => latestSummary && copyText(latestSummary)
        );

        byId('downloadSummaryBtn').addEventListener(
            'click',
            () => latestSummary &&
                downloadText(
                    'documind-summary.txt',
                    latestSummary
                )
        );

        byId('quizForm').addEventListener(
            'submit',
            generateQuiz
        );

        byId('resetQuizBtn').addEventListener(
            'click',
            resetQuiz
        );

        byId('quizOutput').addEventListener(
            'click',
            event => {
                if (event.target.closest('#submitQuizBtn')) {
                    submitQuiz();
                }
            }
        );

        byId('compareForm').addEventListener(
            'submit',
            compareDocuments
        );

        byId('settingsForm').addEventListener(
            'submit',
            saveSettings
        );

        byId('testConnectionBtn').addEventListener(
            'click',
            async () => {
                const connected = await checkBackend();

                toast(
                    connected
                        ? 'Backend connection successful.'
                        : 'Cannot connect. Check that FastAPI is running and the URL is correct.',
                    connected ? 'success' : 'error'
                );
            }
        );

        byId('clearDataBtn').addEventListener(
            'click',
            clearLocalData
        );

        byId('profileMenuBtn').addEventListener(
            'click',
            () => {
                navigate('settings');
                byId('settingsName').focus();
            }
        );

        window.addEventListener(
            'hashchange',
            handleRoute
        );

        window.matchMedia?.(
            '(prefers-color-scheme: dark)'
        ).addEventListener?.(
            'change',
            () => {
                if (state.settings.theme === 'system') {
                    applyTheme('system');
                }
            }
        );
    }

    function setDocumentView(view) {
        state.documentView = view;

        byId('viewGrid').classList.toggle(
            'active',
            view === 'grid'
        );

        byId('viewList').classList.toggle(
            'active',
            view === 'list'
        );

        byId('viewGrid').setAttribute(
            'aria-pressed',
            String(view === 'grid')
        );

        byId('viewList').setAttribute(
            'aria-pressed',
            String(view === 'list')
        );

        renderDocuments();
    }

    /* ------------------------------ Initialisation ------------------------ */

    async function init() {
        applyTheme(state.settings.theme);

        fillSettings();
        updateStats();

        renderChatHistory();
        renderActiveChat();

        bindEvents();
        handleRoute();
        renderDocuments();
        refreshIcons();

        // Check backend availability without blocking the page.
        checkBackend();

        // If health is reachable, load the document library.
        window.setTimeout(async () => {
            if (await checkBackend()) {
                await loadDocuments(false);
            }
        }, 300);
    }

    // Keep the existing inline HTML onclick handlers working.
    window.newChat = createChat;
    window.openFeature = openFeature;

    if (document.readyState === 'loading') {
        document.addEventListener(
            'DOMContentLoaded',
            init,
            { once: true }
        );
    } else {
        init();
    }

})();


