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
   POST /api/chat               -> request {document_id?, mode: "auto"|"pdf"|"ai", question, history, top_k, show_sources}
                                   response {answer, sources:[{source_id, chunk_id, page, text, title}], supported,
                                             mode, answer_type: "pdf"|"general_ai", fallback_reason}
   POST /api/summarize          -> {summary}
   POST /api/quiz/generate      -> {questions:[{question, options, answer_index, explanation}]}
   DELETE /api/documents/{id}   -> success response

   PDFs and API keys are not stored in localStorage. Chat text and settings
   are stored locally in this browser; use a backend database for account sync.
   ========================================================================== */

(() => {
    'use strict';

    // Safety net: if this file is ever loaded twice (for example by a second
    // <script> tag), a second copy would register every event listener again
    // and each question would be sent twice. Only the first copy may run.
    if (window.__documindLoaded) {
        console.warn('DocuMind: script.js was loaded twice; the second copy is ignored.');
        return;
    }
    window.__documindLoaded = true;

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
        quiz: '/api/quiz/generate'
    };

    const PAGE_META = {
        dashboard: ['Dashboard', 'Your intelligent document workspace'],
        documents: ['My Documents', 'Manage your indexed PDF library'],
        assistant: ['AI Assistant', 'Ask your documents or get general AI answers'],
        summarizer: ['Smart Summarizer', 'Turn long PDFs into concise notes'],
        quiz: ['Quiz Generator', 'Test your understanding with document-based questions'],
        settings: ['Settings', 'Customize your DocuMind workspace']
    };

    // Own-property check: a hash such as #constructor or #toString must not be treated as a page.
    const isPage = page => Object.prototype.hasOwnProperty.call(PAGE_META, page);

    // Response modes. The keys are the exact values the backend accepts.
    const DEFAULT_MODE = 'auto';

    const RESPONSE_MODES = {
        auto: {
            label: 'Auto',
            description: 'Searches your PDF first, adds general AI detail when useful (clearly labelled), and uses general AI alone if the PDF has no relevant answer.',
            placeholder: 'Ask anything. Your PDF is searched first…',
            hint: 'Auto: answers from your PDF when it has evidence, otherwise from general AI (clearly labelled). Verify important information.',
            emptyText: 'Ask a question. DocuMind checks your selected PDF first and tells you when it answers from general AI instead.'
        },
        pdf: {
            label: 'Only PDF',
            description: 'Answers only from the selected PDF, with page citations. Never uses outside knowledge.',
            placeholder: 'Ask anything about your document…',
            hint: 'Only PDF: answers use retrieved passages from your PDF. Verify important information.',
            emptyText: 'Select a document and ask your first question. Answers will cite the source pages.'
        },
        ai: {
            label: 'Only AI',
            description: 'General AI answers. Your PDF is not searched and no PDF is needed.',
            placeholder: 'Ask anything…',
            hint: 'Only AI: general knowledge answers. Your PDF is not used. Verify important information.',
            emptyText: 'Ask any question. Only AI does not search your PDFs, so no PDF is needed.'
        }
    };

    const ANSWER_TYPES = ['pdf', 'general_ai', 'hybrid', 'web', 'pdf_web'];

    const normaliseMode = value =>
        typeof value === 'string' && Object.prototype.hasOwnProperty.call(RESPONSE_MODES, value)
            ? value
            : DEFAULT_MODE;

    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
    const byId = id => document.getElementById(id);

    const state = {
        settings: loadJSON(STORAGE_KEYS.settings, DEFAULT_SETTINGS),
        documents: [],
        chats: loadJSON(STORAGE_KEYS.chats, []),
        activeChatId: safeStorageGet(STORAGE_KEYS.activeChat),
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
        isSendingQuestion: false,
        // Mode shown while no conversation exists yet. New conversations start in Auto.
        draftMode: DEFAULT_MODE,
        dragDepth: 0,
        lastDocumentLoadFailed: false,
        requestControllers: new Set()
    };

    state.settings = {
        ...DEFAULT_SETTINGS,
        ...(state.settings && typeof state.settings === 'object' && !Array.isArray(state.settings)
            ? state.settings
            : {})
    };
    state.settings.theme = ['light', 'dark', 'system'].includes(state.settings.theme)
        ? state.settings.theme
        : DEFAULT_SETTINGS.theme;
    state.settings.name = safeText(state.settings.name, DEFAULT_SETTINGS.name);
    state.settings.apiUrl = safeText(state.settings.apiUrl, DEFAULT_SETTINGS.apiUrl);
    state.settings.topK = Math.min(20, Math.max(1, Number(state.settings.topK) || DEFAULT_SETTINGS.topK));
    state.settings.chunkSize = Math.min(4000, Math.max(100, Number(state.settings.chunkSize) || DEFAULT_SETTINGS.chunkSize));
    state.settings.showSources = state.settings.showSources !== false;

    // Ignore malformed entries left by interrupted writes or older app versions.
    state.chats = Array.isArray(state.chats)
        ? state.chats.filter(chat => chat && typeof chat === 'object' && chat.id != null)
            .map(chat => ({
                id: String(chat.id),
                title: safeText(chat.title, 'New conversation'),
                documentId: chat.documentId == null ? '' : String(chat.documentId),
                // Chats saved before response modes existed start in Auto.
                mode: normaliseMode(chat.mode),
                messages: Array.isArray(chat.messages)
                    ? chat.messages.filter(message => message && typeof message === 'object' && ['user', 'assistant'].includes(message.role))
                        .slice(-40)
                        .map(message => ({
                            role: message.role,
                            content: String(message.content ?? ''),
                            ...(Array.isArray(message.sources) ? { sources: message.sources } : {}),
                            ...(Array.isArray(message.webSources) ? { webSources: normaliseWebSources(message.webSources) } : {}),
                            ...(message.webSearch && typeof message.webSearch === 'object' ? { webSearch: normaliseWebSearch(message.webSearch) } : {}),
                            ...(ANSWER_TYPES.includes(message.answerType) ? { answerType: message.answerType } : {}),
                            ...(typeof message.supported === 'boolean' ? { supported: message.supported } : {}),
                            ...(typeof message.fallbackReason === 'string' ? { fallbackReason: message.fallbackReason } : {}),
                            ...(Object.prototype.hasOwnProperty.call(RESPONSE_MODES, message.mode) ? { mode: message.mode } : {}),
                            ...(message.error ? { error: true } : {})
                        }))
                    : [],
                updatedAt: chat.updatedAt || new Date().toISOString()
            }))
        : [];
    state.chats = state.chats.slice(0, 30);
    if (!state.chats.some(chat => chat.id === state.activeChatId)) {
        state.activeChatId = state.chats[0]?.id || null;
    }

    if (!state.counters || typeof state.counters !== 'object' || Array.isArray(state.counters)) {
        state.counters = { questions: 0, summaries: 0, quizzes: 0 };
    } else {
        state.counters = {
            questions: Math.max(0, Number(state.counters.questions) || 0),
            summaries: Math.max(0, Number(state.counters.summaries) || 0),
            quizzes: Math.max(0, Number(state.counters.quizzes) || 0)
        };
    }

    function safeStorageGet(key) {
        try {
            return localStorage.getItem(key) || null;
        } catch (error) {
            console.warn(`DocuMind: could not read ${key}`, error);
            return null;
        }
    }

    function loadJSON(key, fallback) {
        try {
            const value = localStorage.getItem(key);
            return value ? JSON.parse(value) : (Array.isArray(fallback) ? [...fallback] : { ...fallback });
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

    // Only plain http(s) links may become clickable. Anything else
    // (javascript:, data:, malformed text) returns ''.
    function safeExternalUrl(value) {
        if (typeof value !== 'string' || !value.trim() || value.length > 2048) return '';

        try {
            const parsed = new URL(value.trim());
            if (!['http:', 'https:'].includes(parsed.protocol)) return '';
            if (parsed.username || parsed.password) return '';
            return parsed.href;
        } catch (error) {
            return '';
        }
    }

    // Web sources come from the backend (Gemini grounding metadata) or from
    // saved chats. Every field is re-validated before it is stored or shown.
    function normaliseWebSources(list) {
        if (!Array.isArray(list)) return [];

        return list.slice(0, 10).map((item, index) => {
            if (!item || typeof item !== 'object') return null;

            const url = safeExternalUrl(item.url);
            if (!url) return null;

            return {
                source_id: typeof item.source_id === 'string' && /^W\d+$/.test(item.source_id)
                    ? item.source_id
                    : `W${index + 1}`,
                title: safeText(item.title, '').slice(0, 200),
                url,
                domain: safeText(item.domain, '').slice(0, 120),
                snippets: Array.isArray(item.snippets)
                    ? item.snippets.filter(text => typeof text === 'string' && text.trim())
                        .slice(0, 3).map(text => text.slice(0, 300))
                    : [],
                cited: item.cited === true
            };
        }).filter(Boolean);
    }

    function normaliseWebSearch(info) {
        const status = ['used', 'no_sources', 'failed', 'not_used'].includes(info?.status)
            ? info.status
            : 'not_used';

        return {
            status,
            queries: Array.isArray(info?.queries)
                ? info.queries.filter(query => typeof query === 'string' && query.trim())
                    .slice(0, 8).map(query => query.slice(0, 200))
                : []
        };
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
            status: safeText(source.status, 'processing'),
            pages: source.pages ?? source.page_count ?? null
        };
    }

    function apiURL(path) {
        const base = String(
            state.settings.apiUrl || DEFAULT_SETTINGS.apiUrl
        ).trim().replace(/\/+$/, '');

        return `${base}${path.startsWith('/') ? path : `/${path}`}`;
    }

    function formatApiError(payload, status) {
        const raw = typeof payload === 'string'
            ? payload
            : payload?.detail ?? payload?.message ?? payload?.error;

        if (Array.isArray(raw)) {
            return raw.map(item => {
                if (typeof item === 'string') return item;
                const location = Array.isArray(item?.loc)
                    ? item.loc.join(' → ')
                    : '';
                const message = item?.msg || item?.message || 'Invalid request';
                return location ? `${location}: ${message}` : message;
            }).join('; ');
        }

        if (raw && typeof raw === 'object') {
            return raw.message || raw.msg || JSON.stringify(raw);
        }

        return raw ? String(raw) : `Request failed (${status})`;
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
                throw new Error(formatApiError(payload, response.status));
            }

            return payload;

        } catch (error) {
            if (error.name === 'AbortError') {
                throw new Error('The request timed out or was cancelled. Please try again.');
            }
            if (error instanceof TypeError) {
                throw new Error('Cannot reach the backend. Check that FastAPI is running and the API URL is correct.');
            }
            throw error;
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

        if (loading && !button.dataset.originalAriaLabel) {
            button.dataset.originalAriaLabel = button.getAttribute('aria-label') || button.textContent.trim();
        }
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
        if (!isPage(page)) {
            page = 'dashboard';
        }

        // Single source of truth for CSS-driven visibility (see style.css).
        // The inline script in <head> sets the same attribute before first paint.
        document.documentElement.dataset.route = page;

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
            history.pushState({ page }, '', `#${page}`);
        }

        closeSidebar();

        document.title = `${title} | DocuMind`;

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
        if (page === 'mainContent') {
            byId('mainContent')?.focus();
            return;
        }
        navigate(isPage(page) ? page : 'dashboard', false);
    }


    function openFeature(feature) {
        const pages = {
            chat: 'assistant',
            summary: 'summarizer',
            quiz: 'quiz'
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
        el.classList.remove('success', 'error', 'warning');

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

        // The progress bar lives on the Dashboard, so tell the user the upload
        // has started when it was triggered from any other page.
        if (document.documentElement.dataset.route !== 'dashboard') {
            toast(`Uploading ${file.name}…`, 'info');
        }

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

            // Backend duplicate check (SHA-256 of the file contents): nothing
            // was created, so show a warning, keep the document list as it is
            // and let the user upload another PDF.
            if (xhr.status === 409 && payload?.duplicate) {
                const existingName = payload.existing_document?.filename;
                const duplicateMessage = existingName
                    ? `This PDF has already been uploaded as “${existingName}”.`
                    : 'This PDF has already been uploaded.';

                setFileStatus(
                    `Duplicate PDF: ${duplicateMessage} No duplicate document was created.`,
                    'warning'
                );
                toast(`Duplicate PDF — ${duplicateMessage}`, 'warning', 6000);

                resetUploadProgress();
                byId('pdfInput').value = '';

                return;
            }

            if (xhr.status < 200 || xhr.status >= 300) {
                const message = formatApiError(payload, xhr.status);

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
                setFileStatus(`${doc.name} uploaded. Waiting for indexing to finish…`);
                toast('PDF uploaded. Checking indexing status…', 'info');

                await loadDocuments(false);
                let refreshed = state.documents.find(item => item.id === doc.id) || doc;
                for (let attempt = 0; attempt < 5 && !isDocumentReady(refreshed); attempt += 1) {
                    await new Promise(resolve => window.setTimeout(resolve, 2000));
                    await loadDocuments(false);
                    refreshed = state.documents.find(item => item.id === doc.id) || refreshed;
                }
                setFileStatus(isDocumentReady(refreshed) ? `${refreshed.name} is ready to use.` : `${refreshed.name} was uploaded and is still processing.`, isDocumentReady(refreshed) ? 'success' : '');
                // Dashboard uploads keep their original redirect; uploads started
                // from other pages (My Documents, Assistant, Summarizer, Quiz) stay put.
                if (document.documentElement.dataset.route === 'dashboard') {
                    navigate('documents');
                }
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
        formData.append('chunk_size', String(state.settings.chunkSize));

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

    function isDocumentReady(doc) {
        const status = String(doc.status || '').toLowerCase();
        return ['ready', 'indexed', 'complete', 'completed', 'processed', 'success'].includes(status);
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
                    <i data-lucide="file-text"></i>
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
                .sort((a, b) => new Date(b.uploadedAt || 0) - new Date(a.uploadedAt || 0))
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
            'quizDocumentSelect'
        ];

        ids.forEach(id => {
            const select = byId(id);

            if (!select) return;

            const previous = select.value;

            const first = select.options[0]?.outerHTML ||
                '<option value="">Select a document…</option>';

            select.innerHTML = first +
                state.documents.map(doc => `
                    <option value="${escapeHTML(doc.id)}" ${isDocumentReady(doc) ? '' : 'disabled'}>
                        ${escapeHTML(doc.name)}${isDocumentReady(doc) ? '' : ` (${escapeHTML(doc.status)})`}
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
            state.lastDocumentLoadFailed = false;

            renderDocuments();

            if (showMessage) {
                toast(
                    `${state.documents.length} document(s) loaded.`,
                    'success'
                );
            }

        } catch (error) {
            state.lastDocumentLoadFailed = true;
            // Keep the last successful list instead of replacing it with an empty library.
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
        // Every explicit New Chat action creates a distinct conversation.
        const chat = {
            id: makeId(),
            title: 'New conversation',
            documentId: '',
            mode: DEFAULT_MODE,
            messages: [],
            updatedAt: new Date().toISOString()
        };

        state.chats.unshift(chat);
        state.activeChatId = chat.id;
        state.draftMode = DEFAULT_MODE;

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

    /* ------------------------- Response mode ------------------------- */

    // The mode shown in the selector is the mode of the open conversation
    // (or the draft mode while no conversation exists). sendQuestion() reads
    // it from here, so what is displayed is what is sent.
    function currentMode() {
        const chat = activeChat();
        return normaliseMode(chat ? chat.mode : state.draftMode);
    }

    function setMode(value) {
        const mode = normaliseMode(value);
        const chat = activeChat();

        if (chat) {
            chat.mode = mode;
            persistChats();
        } else {
            state.draftMode = mode;
        }

        updateModeUI();
    }

    function updateModeUI() {
        const mode = currentMode();
        const meta = RESPONSE_MODES[mode];
        const group = byId('chatModeGroup');

        if (group) {
            $$('input[name="chatMode"]', group).forEach(input => {
                input.checked = input.value === mode;
                // The mode cannot change while a question is being answered.
                input.disabled = state.isSendingQuestion;
            });
        }

        byId('chatModeBar')?.setAttribute('data-mode', mode);
        byId('chatContext')?.classList.toggle('mode-ai', mode === 'ai');

        // Only AI never uses a PDF, so the document selector is disabled.
        // The chosen document is kept and becomes usable again in Auto / Only PDF.
        const documentSelect = byId('chatDocumentSelect');
        if (documentSelect) {
            documentSelect.disabled = mode === 'ai';
            documentSelect.title = mode === 'ai'
                ? 'Not used in Only AI mode. Switch to Auto or Only PDF to choose a document.'
                : '';
        }

        const description = byId('chatModeDescription');
        if (description) description.textContent = meta.description;

        const input = byId('chatInput');
        if (input) input.placeholder = meta.placeholder;

        const hint = byId('chatHint');
        if (hint) hint.textContent = meta.hint;
    }

    function selectChat(id) {
        // Resolve the clicked row by its own ID; never fall back to another chat.
        const selected = state.chats.find(chat => chat.id === id);
        if (!selected) {
            toast('This conversation could not be found. Refresh the page and try again.', 'error');
            return;
        }

        state.activeChatId = selected.id;
        persistChats();
        renderActiveChat();
        navigate('assistant');
    }

    function persistChats() {
        // Keep browser storage bounded; durable account history belongs in the backend database.
        //
        // The list is trimmed IN PLACE. Earlier versions replaced every chat with a copy
        // here. A question that was still waiting for its answer kept a reference to the
        // old object, so when the answer arrived it was added to a chat that was no longer
        // in state.chats: the answer was lost and the user message appeared to be sent twice.
        if (state.chats.length > 30) {
            state.chats.length = 30;
        }

        state.chats.forEach(chat => {
            if (!Array.isArray(chat.messages)) {
                chat.messages = [];
            }

            if (chat.messages.length > 40) {
                chat.messages.splice(0, chat.messages.length - 40);
            }
        });

        if (!state.chats.some(chat => chat.id === state.activeChatId)) {
            state.activeChatId = state.chats[0]?.id || null;
        }
        saveJSON(STORAGE_KEYS.chats, state.chats);

        if (state.activeChatId) {
            try { localStorage.setItem(STORAGE_KEYS.activeChat, state.activeChatId); }
            catch (error) { console.warn('Could not save active chat selection.', error); }
        } else {
            try { localStorage.removeItem(STORAGE_KEYS.activeChat); } catch (_) { /* storage may be disabled */ }
        }
    }

    function renderChatHistory() {
        const list = byId('chatHistory');
        if (!list) return;

        if (!state.chats.length) {
            list.innerHTML = `
                <li class="chat-history-empty" id="chatHistoryEmpty">
                    No conversations yet.
                </li>
            `;
            return;
        }

        list.innerHTML = state.chats.slice(0, 30).map(chat => `
            <li class="chat-history-row">
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
                <button
                    type="button"
                    class="chat-history-delete"
                    data-action="delete-chat"
                    data-id="${escapeHTML(chat.id)}"
                    aria-label="Delete conversation: ${escapeHTML(chat.title)}"
                    title="Delete conversation"
                >
                    <i data-lucide="trash-2" aria-hidden="true"></i>
                </button>
            </li>
        `).join('');

        refreshIcons();
    }

    function deleteChat(id) {
        const chat = state.chats.find(item => item.id === id);
        if (!chat) return;

        const confirmed = window.confirm(
            `Delete "${chat.title || 'New conversation'}"? This chat history cannot be recovered.`
        );
        if (!confirmed) return;

        state.chats = state.chats.filter(item => item.id !== id);

        if (state.activeChatId === id) {
            state.activeChatId = state.chats[0]?.id || null;
        }

        if (!state.chats.length) state.draftMode = DEFAULT_MODE;

        persistChats();
        renderChatHistory();
        renderActiveChat();
        toast('Conversation deleted.', 'success');
    }

    function clearChatHistory() {
        if (!state.chats.length) {
            toast('There is no chat history to delete.', 'info');
            return;
        }

        const confirmed = window.confirm(
            'Delete all chat history? This will permanently remove all saved conversations from this browser. Uploaded PDFs and settings will not be affected.'
        );
        if (!confirmed) return;

        state.chats = [];
        state.activeChatId = null;
        state.draftMode = DEFAULT_MODE;
        persistChats();
        renderChatHistory();
        renderActiveChat();
        toast('All chat history deleted.', 'success');
    }

    function renderSafeMarkdown(target, source, sources = []) {
        // Escape untrusted content before adding the limited Markdown and citation markup.
        const sourceIds = new Set(
            (Array.isArray(sources) ? sources : []).map((item, index) => {
                const id = typeof item.source_id === 'string' ? item.source_id : `S${index + 1}`;
                return /^[SW]\d+$/.test(id) ? id : null;
            }).filter(Boolean)
        );
        // Line-by-line rendering. The model often puts a heading, bullets and plain
        // sentences in one block without blank lines between them, so every line is
        // classified on its own instead of classifying whole blocks.
        const html = [];
        let listType = null;
        let paragraph = [];

        const flushParagraph = () => {
            if (!paragraph.length) return;
            html.push(`<p>${paragraph.map(line => formatInlineMarkdown(line, sourceIds)).join('<br>')}</p>`);
            paragraph = [];
        };

        const closeList = () => {
            if (!listType) return;
            html.push(`</${listType}>`);
            listType = null;
        };

        escapeHTML(source || '').split('\n').forEach(rawLine => {
            const line = rawLine.trim();

            if (!line) {
                flushParagraph();
                closeList();
                return;
            }

            const heading = line.match(/^(#{1,3})\s+(.+)$/);
            const bullet = line.match(/^[-*\u2022]\s+(.+)$/);
            const numbered = line.match(/^\d+\.\s+(.+)$/);

            if (heading) {
                flushParagraph();
                closeList();
                html.push(`<h${heading[1].length}>${formatInlineMarkdown(heading[2], sourceIds)}</h${heading[1].length}>`);
                return;
            }

            if (bullet || numbered) {
                flushParagraph();

                const type = bullet ? 'ul' : 'ol';

                if (listType !== type) {
                    closeList();
                    html.push(`<${type}>`);
                    listType = type;
                }

                html.push(`<li>${formatInlineMarkdown((bullet || numbered)[1], sourceIds)}</li>`);
                return;
            }

            closeList();
            paragraph.push(line);
        });

        flushParagraph();
        closeList();

        target.innerHTML = html.join('');
    }

    function formatInlineMarkdown(text, sourceIds = new Set()) {
        const formatted = text
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
            .replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Handles single citations such as [S1] and grouped citations such as
        // [S1, S3] (saved by older versions of the backend). Every label that has a
        // matching source becomes a button; unknown labels stay as plain text.
        // [W1] markers point to web sources and get a web-styled button.
        return formatted.replace(
            /\[\s*([SW]\d+(?:\s*[,;]\s*[SW]\d+)*)\s*\]/g,
            (match, group) => {
                const ids = group.match(/[SW]\d+/g) || [];

                if (!ids.some(sourceId => sourceIds.has(sourceId))) return match;

                return ids.map(sourceId => (
                    sourceIds.has(sourceId)
                        ? `<button type="button" class="citation${sourceId.startsWith('W') ? ' citation--web' : ''}" data-action="open-source" data-source-id="${sourceId}" aria-label="Open ${sourceId.startsWith('W') ? 'web ' : ''}source ${sourceId}">${sourceId}</button>`
                        : `[${sourceId}]`
                )).join(' ');
            }
        );
    }

    // Badge that tells the user where an answer came from. Built with
    // textContent only: nothing from the model is inserted as HTML.
    function buildAnswerMeta(message) {
        let badgeText = '';
        let badgeClass = '';
        let icon = '';
        let note = '';

        if (message.answerType === 'pdf') {
            if (message.supported === false) {
                badgeText = 'Not Found in PDF';
                badgeClass = 'answer-badge--not-found';
                icon = 'file-x';
            } else {
                badgeText = 'PDF-Based Answer';
                badgeClass = 'answer-badge--pdf';
                icon = 'file-text';
            }
        } else if (message.answerType === 'hybrid') {
            badgeText = 'PDF + General AI Answer';
            badgeClass = 'answer-badge--hybrid';
            icon = 'layers';
            note = 'Cited statements come from your PDF. The section marked "general AI" does not.';
        } else if (message.answerType === 'web') {
            badgeText = 'Web-Grounded Answer';
            badgeClass = 'answer-badge--web';
            icon = 'globe';
            note = 'Based on Google Search results, not on your PDF. Check the web sources.';
        } else if (message.answerType === 'pdf_web') {
            badgeText = 'PDF + Web Answer';
            badgeClass = 'answer-badge--hybrid';
            icon = 'layers';
            note = 'Cited statements come from your PDF. The section marked "web" comes from Google Search, not from your PDF.';
        } else if (message.answerType === 'general_ai') {
            badgeText = 'General AI Answer';
            badgeClass = 'answer-badge--general';
            icon = 'sparkles';

            const notes = {
                no_pdf_selected: 'No PDF was selected, so this is not based on a document.',
                insufficient_evidence: 'Your PDF did not contain enough relevant information, so this is not from the PDF.',
                unverified_evidence: 'The PDF evidence could not be confirmed for this question, so this is not from the PDF.'
            };

            note = notes[message.fallbackReason] ||
                (message.mode === 'ai' ? 'Your PDF was not searched.' : 'Not based on your PDF.');
        } else {
            // Messages saved by older versions have no answer type.
            return null;
        }

        // Honest status when current information was wanted but not retrieved.
        const webStatus = message.webSearch?.status;

        if (webStatus === 'failed') {
            note = `${note} Live web search was unavailable, so this answer may be out of date.`.trim();
        } else if (webStatus === 'no_sources') {
            note = `${note} No web sources were returned, so this answer may not reflect the latest information.`.trim();
        }

        const meta = document.createElement('div');
        meta.className = 'answer-meta';

        const badge = document.createElement('span');
        badge.className = `answer-badge ${badgeClass}`;

        const badgeIcon = document.createElement('i');
        badgeIcon.setAttribute('data-lucide', icon);
        badgeIcon.setAttribute('aria-hidden', 'true');

        const badgeLabel = document.createElement('span');
        badgeLabel.textContent = badgeText;

        badge.append(badgeIcon, badgeLabel);
        meta.appendChild(badge);

        if (note) {
            const noteElement = document.createElement('span');
            noteElement.className = 'answer-note';
            noteElement.textContent = note;
            meta.appendChild(noteElement);
        }

        return meta;
    }

    // Sources panel content for one assistant message.
    function renderSourcesForMessage(message) {
        if (message?.answerType === 'general_ai') {
            renderSources([], 'General AI answers do not use PDF sources.');
            return;
        }

        renderSources(
            message?.sources || [],
            undefined,
            message?.webSources || [],
            message?.webSearch || null
        );
    }

    function renderMessage(message, messageIndex = -1) {
        const isUser = message.role === 'user';

        const wrapper = document.createElement('div');

        wrapper.className =
            `message ${isUser ? 'user' : 'assistant'}` +
            `${message.error ? ' error' : ''}`;
        wrapper.dataset.messageIndex = String(messageIndex);

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';

        avatar.textContent = isUser
            ? (state.settings.name?.[0] || 'Y').toUpperCase()
            : 'AI';

        const content = document.createElement('div');
        content.className = 'message-content';

        const isGeneral = message.answerType === 'general_ai';

        if (!isUser && !message.error) {
            const meta = buildAnswerMeta(message);
            if (meta) content.appendChild(meta);
        }

        const body = document.createElement('div');
        body.className = 'message-body';
        // A general AI answer has no PDF sources, so no citation buttons are made.
        // Citation buttons exist for PDF sources [S1] and web sources [W1].
        renderSafeMarkdown(
            body,
            message.content || '',
            isGeneral ? [] : [...(message.sources || []), ...(message.webSources || [])]
        );
        content.appendChild(body);

        if (
            !isGeneral &&
            Array.isArray(message.sources) &&
            message.sources.length &&
            state.settings.showSources
        ) {
            const sourceLine = document.createElement('div');

            sourceLine.className = 'message-sources';

            const uniquePages = [...new Set(message.sources.map(source =>
                String(source.page ?? source.page_number ?? '?')
            ))];

            sourceLine.textContent = uniquePages.map(page =>
                `Page ${page}`
            ).join(' · ');

            content.appendChild(sourceLine);
        }

        if (
            !isGeneral &&
            Array.isArray(message.webSources) &&
            message.webSources.length &&
            state.settings.showSources
        ) {
            const webLine = document.createElement('div');
            webLine.className = 'message-sources';

            const labels = [...new Set(message.webSources.map(source =>
                source.domain || source.title || 'Web source'
            ))].slice(0, 4);

            webLine.textContent = `Web: ${labels.join(' · ')}`;
            content.appendChild(webLine);
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
                    <p>${escapeHTML(RESPONSE_MODES[currentMode()].emptyText)}</p>
                </div>
            `;
        } else {
            messages.replaceChildren(
                ...chat.messages.map((message, index) => renderMessage(message, index))
            );

            messages.scrollTop = messages.scrollHeight;
        }

        // Each conversation owns its document selection. Do not leave the
        // previous chat's document selected when the newly opened chat has none.
        const documentSelect = byId('chatDocumentSelect');
        if (documentSelect) {
            const selectedDocumentId = chat?.documentId || '';
            documentSelect.value = state.documents.some(
                doc => doc.id === selectedDocumentId
            ) ? selectedDocumentId : '';
        }

        renderChatHistory();
        updateModeUI();
        const lastAssistant = [...(chat?.messages || [])].reverse().find(message => message.role === 'assistant');
        renderSourcesForMessage(lastAssistant);
        refreshIcons();
    }

    function renderWebSourceCard(source) {
        const title = source.title || source.domain || 'Web source';

        const snippets = (source.snippets || []).slice(0, 2).map(text => `
                    <p class="source-text">&ldquo;${escapeHTML(text)}&rdquo;</p>
                `).join('');

        // The link was validated by safeExternalUrl(); it still gets escaped.
        return `
                <article class="source-card source-card--web" data-source-id="${escapeHTML(source.source_id)}">
                    <div class="source-header">
                        <strong class="source-title">
                            ${escapeHTML(title)}
                        </strong>

                        <span class="source-page source-page--web">
                            ${escapeHTML(source.source_id)}
                        </span>
                    </div>

                    <a class="source-link" href="${escapeHTML(source.url)}" target="_blank" rel="noopener noreferrer nofollow">
                        ${escapeHTML(source.domain || 'Open source')} &#8599;
                    </a>
                    ${snippets}
                </article>
            `;
    }

    function renderSources(
        sources,
        emptyMessage = 'Retrieved passages will appear here after you ask a question.',
        webSources = [],
        webSearch = null
    ) {
        const list = byId('sourcesList');

        const safeWebSources = normaliseWebSources(webSources);
        const hasPdfSources = Array.isArray(sources) && sources.length > 0;

        if (
            !state.settings.showSources ||
            (!hasPdfSources && !safeWebSources.length)
        ) {
            list.innerHTML = `
                <div class="empty-state compact">
                    <div class="empty-icon">
                        <i data-lucide="quote"></i>
                    </div>
                    <p>
                        ${escapeHTML(emptyMessage)}
                    </p>
                </div>
            `;

            refreshIcons();
            return;
        }

        const pdfHtml = (hasPdfSources ? sources : []).map((source, index) => {
            const sourceId =
                typeof source.source_id === 'string' && /^S\d+$/.test(source.source_id)
                    ? source.source_id
                    : `S${index + 1}`;

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
                <article class="source-card" data-source-id="${escapeHTML(sourceId)}">
                    <div class="source-header">
                        <strong class="source-title">
                            ${escapeHTML(title)} (${escapeHTML(sourceId)})
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

        // Web sources are kept apart from PDF sources. With only PDF
        // sources the panel looks exactly as before.
        if (!safeWebSources.length) {
            list.innerHTML = pdfHtml;
            return;
        }

        const queries = normaliseWebSearch(webSearch).queries;
        const queryLine = queries.length
            ? `<p class="source-queries">Searched for: ${escapeHTML(queries.join(' · '))}</p>`
            : '';

        const pdfSection = hasPdfSources
            ? `<h4 class="sources-section-title">PDF Sources</h4>${pdfHtml}`
            : '';

        list.innerHTML = `
            ${pdfSection}
            <h4 class="sources-section-title">Web Sources</h4>
            ${queryLine}
            ${safeWebSources.map(renderWebSourceCard).join('')}
        `;
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

        if (state.isSendingQuestion || !question) return;

        // Read the mode once. The same value is sent to the backend and stored
        // with the answer, so the displayed mode is always the mode that was used.
        const mode = currentMode();

        // Only PDF needs a document. Auto works without one (general AI),
        // and Only AI never uses one.
        if (mode === 'pdf' && !documentId) {
            toast(
                'Only PDF mode needs a PDF. Select or upload an indexed PDF, or switch to Auto or Only AI.',
                'warning',
                6000
            );

            byId('chatDocumentSelect').focus();
            return;
        }

        const selectedDoc = state.documents.find(doc => doc.id === documentId);
        if (mode !== 'ai' && selectedDoc && !isDocumentReady(selectedDoc)) {
            toast('This PDF is still processing. Please wait until indexing is complete.', 'warning');
            return;
        }

        let chat = activeChat();

        if (!chat) {
            chat = {
                id: makeId(),
                title: question.slice(0, 42),
                documentId,
                mode,
                messages: [],
                updatedAt: new Date().toISOString()
            };

            state.chats.unshift(chat);
            state.activeChatId = chat.id;
        }

        const requestChatId = chat.id;
        chat.documentId = documentId;

        if (!chat.messages.length) {
            chat.title = question.slice(0, 42);
        }

        chat.messages.push({
            role: 'user',
            content: question
        });

        chat.updatedAt = new Date().toISOString();

        state.isSendingQuestion = true;
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
                    // The first question after the server starts can be slow
                    // (the embedding model is loaded on first use).
                    timeout: 90000,
                    body: JSON.stringify({
                        // Only AI never sends a document: the PDF is not used at all.
                        document_id: mode === 'ai' ? null : (documentId || null),
                        mode,
                        conversation_id: chat.id,
                        // Error messages are not real answers; never send them as context.
                        history: chat.messages
                            .slice(0, -1)
                            .filter(message => !message.error)
                            .slice(-8)
                            .map(message => ({ role: message.role, content: message.content })),
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

            // What the answer really was (it can differ from the mode in Auto).
            const answerType = ANSWER_TYPES.includes(result.answer_type)
                ? result.answer_type
                : null;

            // A general AI answer never carries PDF sources.
            const sources = answerType === 'general_ai'
                ? []
                : (Array.isArray(result.sources)
                    ? result.sources
                    : (result.citations || []));

            // Web sources are re-validated here; links that are not plain
            // http(s) URLs are dropped.
            const webSources = (answerType === 'web' || answerType === 'pdf_web')
                ? normaliseWebSources(result.web_sources)
                : [];

            const webSearch = result.web_search && typeof result.web_search === 'object'
                ? normaliseWebSearch(result.web_search)
                : null;

            const assistantMessage = {
                role: 'assistant',
                content: answer,
                sources,
                ...(webSources.length ? { webSources } : {}),
                ...(webSearch && webSearch.status !== 'not_used' ? { webSearch } : {}),
                mode,
                ...(answerType ? { answerType } : {}),
                ...(typeof result.supported === 'boolean' ? { supported: result.supported } : {}),
                ...(typeof result.fallback_reason === 'string' ? { fallbackReason: result.fallback_reason } : {})
            };

            chat.messages.push(assistantMessage);

            if (state.activeChatId === requestChatId) renderSourcesForMessage(assistantMessage);

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

            if (state.activeChatId === requestChatId) renderSources([]);

        } finally {
            state.isSendingQuestion = false;
            if (state.activeChatId === requestChatId) byId('typingMessage')?.remove();
            updateModeUI();

            setButtonLoading(
                byId('chatSendBtn'),
                false,
                'Send question'
            );

            persistChats();
            if (state.activeChatId === requestChatId) {
                renderActiveChat();
            }
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
            renderSafeMarkdown(content, latestSummary);

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

        // Allow the browser to start the download before releasing the object URL.
        window.setTimeout(() => URL.revokeObjectURL(url), 1000);
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

        $$('.quiz-feedback, .quiz-explanation, .quiz-source').forEach(el => {
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

            explanation: safeText(q.explanation, ''),
            sourcePages: Array.isArray(q.source_pages)
                ? [...new Set(q.source_pages.map(Number).filter(page => Number.isInteger(page) && page > 0))]
                : []
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

            if (question.sourcePages.length) {
                const source = document.createElement('p');
                source.className = 'quiz-source';
                source.textContent = `PDF source: Page${question.sourcePages.length > 1 ? 's' : ''} ${question.sourcePages.join(', ')}`;
                section.appendChild(source);
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
            ((state.settings.name || 'A').trim()[0] || 'A').toUpperCase();
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

        const settingsSaved = saveJSON(
            STORAGE_KEYS.settings,
            state.settings
        );

        fillSettings();
        applyTheme(state.settings.theme);

        toast(
            settingsSaved
                ? 'Settings saved.'
                : 'Settings applied for this session, but could not be saved in browser storage.',
            settingsSaved ? 'success' : 'warning'
        );

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
        state.draftMode = DEFAULT_MODE;

        state.counters = {
            questions: 0,
            summaries: 0,
            quizzes: 0
        };

        const chatsSaved = saveJSON(STORAGE_KEYS.chats, []);
        const countersSaved = saveJSON(STORAGE_KEYS.counters, state.counters);

        let activeChatCleared = true;
        try {
            localStorage.removeItem(STORAGE_KEYS.activeChat);
        } catch (error) {
            activeChatCleared = false;
            console.warn('Could not clear active chat selection.', error);
        }

        // Keep API URL and theme so the user does not lose setup.
        state.settings = {
            ...DEFAULT_SETTINGS,
            apiUrl: state.settings.apiUrl,
            theme: state.settings.theme
        };

        const settingsSaved = saveJSON(
            STORAGE_KEYS.settings,
            state.settings
        );

        fillSettings();
        renderChatHistory();
        renderActiveChat();
        updateStats();

        const dataSaved = chatsSaved && countersSaved && activeChatCleared && settingsSaved;
        toast(
            dataSaved
                ? 'Local chats and usage data cleared.'
                : 'Local data was cleared for this session, but browser storage could not save every change. Previous data may return after refresh.',
            dataSaved ? 'success' : 'warning'
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
        // Centralised event delegation for navigation and actions.

        document.addEventListener('click', event => {
            const citation = event.target.closest('[data-action="open-source"]');

            if (citation) {
                const chat = activeChat();
                const messageElement = citation.closest('.message');
                const messageIndex = Number(messageElement?.dataset.messageIndex);
                const message = Number.isInteger(messageIndex)
                    ? chat?.messages?.[messageIndex]
                    : null;
                const sourceId = citation.dataset.sourceId;

                if (!message || (!Array.isArray(message.sources) && !Array.isArray(message.webSources))) {
                    toast('The source for this citation is not available.', 'warning');
                    return;
                }

                renderSources(message.sources || [], undefined, message.webSources || [], message.webSearch || null);
                const sourceCard = Array.from(
                    byId('sourcesList').querySelectorAll('.source-card')
                ).find(card => card.dataset.sourceId === sourceId);

                if (!sourceCard) {
                    toast('The cited source could not be found.', 'warning');
                    return;
                }

                sourceCard.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'nearest' });
                sourceCard.classList.add('active');
                window.setTimeout(() => sourceCard.classList.remove('active'), 2200);
                return;
            }

            const pageLink = event.target.closest('[data-page]');

            if (pageLink) {
                event.preventDefault();
                navigate(pageLink.dataset.page);
                return;
            }

            const feature = event.target.closest('[data-feature]');
            if (feature) {
                openFeature(feature.dataset.feature);
                return;
            }

            const action = event.target.closest('[data-action]');

            if (!action) return;

            const id = action.dataset.id;

            switch (action.dataset.action) {
                case 'browse':
                    byId('pdfInput').click();
                    break;

                case 'ask': {
                    const doc = state.documents.find(item => item.id === id);
                    if (!doc || !isDocumentReady(doc)) {
                        toast('This document is not ready yet.', 'warning');
                        break;
                    }
                    let chat = activeChat();
                    if (!chat || chat.messages.length) {
                        chat = { id: makeId(), title: `Questions about ${doc.name}`.slice(0, 42), documentId: id, mode: DEFAULT_MODE, messages: [], updatedAt: new Date().toISOString() };
                        state.chats.unshift(chat);
                        state.activeChatId = chat.id;
                    } else {
                        chat.documentId = id;
                        // Asking about a document means the PDF should be used.
                        if (chat.mode === 'ai') chat.mode = DEFAULT_MODE;
                    }
                    persistChats();
                    navigate('assistant');
                    byId('chatDocumentSelect').value = id;
                    renderActiveChat();
                    break;
                }

                case 'delete':
                    deleteDocument(id);
                    break;

                case 'select-chat':
                    event.preventDefault();
                    event.stopPropagation();
                    selectChat(id);
                    break;

                case 'delete-chat':
                    deleteChat(id);
                    break;

                case 'clear-chat-history':
                    clearChatHistory();
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
            () => byId('pdfInput').click()
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

        window.addEventListener('dragover', event => event.preventDefault());
        window.addEventListener('drop', event => {
            event.preventDefault();
            if (!dropZone.contains(event.target)) {
                toast('Drop a PDF inside the upload area.', 'info');
            }
        });
        dropZone.addEventListener('dragenter', event => {
            event.preventDefault();
            state.dragDepth += 1;
            dropZone.classList.add('drag-over');
        });
        dropZone.addEventListener('dragover', event => event.preventDefault());
        dropZone.addEventListener('dragleave', event => {
            event.preventDefault();
            state.dragDepth = Math.max(0, state.dragDepth - 1);
            if (state.dragDepth === 0) dropZone.classList.remove('drag-over');
        });
        dropZone.addEventListener('drop', event => {
            event.preventDefault();
            state.dragDepth = 0;
            dropZone.classList.remove('drag-over');
            const files = Array.from(event.dataTransfer?.files || []);
            if (files.length > 1) toast('Only one PDF can be uploaded at a time. The first file will be used.', 'info');
            if (files[0]) uploadPDF(files[0]);
        });




        byId('documentsSearch').addEventListener(
            'input',
            event => {
                state.searchTerm = event.target.value.trim().toLowerCase();
                byId('globalSearch').value = event.target.value;
                renderDocuments();
            }
        );

        byId('documentsSort').addEventListener(
            'change',
            renderDocuments
        );

        byId('globalSearch').addEventListener('change', event => {
            const query = event.target.value.trim();
            byId('documentsSearch').value = query;
            state.searchTerm = query.toLowerCase();
            if (query) navigate('documents');
            renderDocuments();
        });
        byId('globalSearch').addEventListener('keydown', event => {
            if (event.key === 'Enter') {
                event.preventDefault();
                byId('globalSearch').dispatchEvent(new Event('change', { bubbles: true }));
            }
            if (event.key === 'Escape') {
                event.target.value = '';
                byId('documentsSearch').value = '';
                state.searchTerm = '';
                renderDocuments();
            }
        });

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

        byId('chatModeGroup')?.addEventListener(
            'change',
            event => {
                const target = event.target;

                if (
                    target instanceof HTMLInputElement &&
                    target.name === 'chatMode' &&
                    !state.isSendingQuestion
                ) {
                    setMode(target.value);
                }
            }
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
                    !event.shiftKey &&
                    !event.isComposing &&
                    !state.isSendingQuestion
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

        window.addEventListener('hashchange', handleRoute);
        window.addEventListener('popstate', handleRoute);

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

        // The correct page is already visible (html[data-route] was set before first
        // paint). This syncs titles, aria-current and the .active/[hidden] mirror state.
        handleRoute();

        renderDocuments();
        refreshIcons();

        // One initial health check; load documents only after a successful response.
        if (await checkBackend()) await loadDocuments(false);

        // Retry on tab focus so a backend started later can reconnect.
        window.addEventListener('focus', async () => {
            if (!state.healthChecked || !byId('backendStatus').classList.contains('connected')) {
                if (await checkBackend()) await loadDocuments(false);
            }
        });
    }


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