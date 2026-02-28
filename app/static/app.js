/**
 * Multilingual RAG System — Frontend Application
 *
 * Handles: tab switching, PDF upload, Q&A chat, document management,
 * and ServiceNow CMDB lookup.
 */

const API = '';  // Same origin

// ── Tab Navigation ───────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`panel-${btn.dataset.tab}`).classList.add('active');

        // Auto-refresh documents when switching to that tab
        if (btn.dataset.tab === 'documents') loadDocuments();
    });
});

// ── Toast Notifications ──────────────────────────────────────────────

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = {
        success: '✓',
        error: '✕',
        info: 'ℹ',
    };
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${message}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = '0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ── File Upload ──────────────────────────────────────────────────────

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const uploadProgress = document.getElementById('upload-progress');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');

// Drag & drop visual feedback
uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
});

fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleUpload(fileInput.files[0]);
});

async function handleUpload(file) {
    if (file.type !== 'application/pdf') {
        showToast('Only PDF files are accepted.', 'error');
        return;
    }

    if (file.size > 50 * 1024 * 1024) {
        showToast('File exceeds 50 MB limit.', 'error');
        return;
    }

    // Show progress
    uploadProgress.classList.add('show');
    progressFill.style.width = '0%';
    progressText.textContent = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Simulate progress since fetch doesn't support upload progress natively
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress = Math.min(progress + Math.random() * 15, 90);
            progressFill.style.width = `${progress}%`;
        }, 300);

        progressText.textContent = 'Processing & embedding document...';

        const resp = await fetch(`${API}/upload`, {
            method: 'POST',
            body: formData,
        });

        clearInterval(progressInterval);

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Upload failed');
        }

        const data = await resp.json();
        progressFill.style.width = '100%';
        progressText.textContent = `✓ ${data.filename} — ${data.total_chunks} chunks indexed (${data.detected_language})`;

        showToast(`Document uploaded! ${data.total_chunks} chunks indexed.`, 'success');
        fileInput.value = '';

        // Reset after 3s
        setTimeout(() => {
            uploadProgress.classList.remove('show');
        }, 3000);
    } catch (err) {
        progressFill.style.width = '0%';
        progressText.textContent = `✕ Error: ${err.message}`;
        showToast(err.message, 'error');
        setTimeout(() => uploadProgress.classList.remove('show'), 4000);
    }
}

// ── Q&A Chat ─────────────────────────────────────────────────────────

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const btnAsk = document.getElementById('btn-ask');

chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askQuestion();
    }
});

async function askQuestion() {
    const question = chatInput.value.trim();
    if (!question) return;

    // Add user bubble
    addBubble(question, 'user');
    chatInput.value = '';
    btnAsk.disabled = true;

    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'typing-indicator';
    typing.innerHTML = '<span></span><span></span><span></span>';
    chatMessages.appendChild(typing);
    scrollChat();

    try {
        const resp = await fetch(`${API}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        typing.remove();

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.detail || 'Query failed');
        }

        const data = await resp.json();

        // Build answer HTML
        let html = escapeHTML(data.answer);

        // Meta info
        html += `<div class="meta">`;
        html += `<span>🌐 ${data.detected_language}</span>`;
        html += `<span>📡 ${data.routed_to === 'servicenow' ? 'ServiceNow' : 'RAG Pipeline'}</span>`;
        html += `</div>`;

        // Sources
        if (data.source_documents && data.source_documents.length > 0) {
            html += `<div class="chat-sources"><details><summary>📄 ${data.source_documents.length} source(s)</summary>`;
            data.source_documents.forEach(src => {
                html += `<span class="source-chip">📄 ${escapeHTML(src.filename)} p.${src.page_number} (${src.score})</span>`;
            });
            html += `</details></div>`;
        }

        addBubble(html, 'assistant', true);
    } catch (err) {
        typing.remove();
        addBubble(`❌ Error: ${escapeHTML(err.message)}`, 'assistant', true);
        showToast(err.message, 'error');
    } finally {
        btnAsk.disabled = false;
        chatInput.focus();
    }
}

function addBubble(content, role, isHTML = false) {
    const bubble = document.createElement('div');
    bubble.className = `chat-bubble ${role}`;
    if (isHTML) {
        bubble.innerHTML = content;
    } else {
        bubble.textContent = content;
    }
    chatMessages.appendChild(bubble);
    scrollChat();
}

function scrollChat() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ── Document Management ──────────────────────────────────────────────

async function loadDocuments() {
    const docList = document.getElementById('doc-list');

    try {
        const resp = await fetch(`${API}/documents`);
        if (!resp.ok) throw new Error('Failed to load documents');

        const data = await resp.json();

        if (!data.documents || data.documents.length === 0) {
            docList.innerHTML = `
        <div class="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <p>No documents indexed yet. Upload a PDF to get started.</p>
        </div>`;
            return;
        }

        docList.innerHTML = data.documents.map(doc => `
      <div class="doc-item" id="doc-${doc.document_id}">
        <div class="doc-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
        </div>
        <div class="doc-info">
          <div class="doc-name">${escapeHTML(doc.filename)}</div>
          <div class="doc-meta">
            <span>🌐 ${doc.detected_language}</span>
            <span>📦 ${doc.chunk_count} chunks</span>
            <span>🕒 ${new Date(doc.upload_timestamp).toLocaleDateString()}</span>
          </div>
        </div>
        <div class="doc-actions">
          <button class="btn btn-danger" onclick="deleteDocument('${doc.document_id}')">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
            </svg>
            Delete
          </button>
        </div>
      </div>
    `).join('');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function deleteDocument(docId) {
    if (!confirm('Delete this document and all its chunks?')) return;

    try {
        const resp = await fetch(`${API}/documents/${docId}`, { method: 'DELETE' });
        if (!resp.ok) throw new Error('Delete failed');

        showToast('Document deleted.', 'success');
        const el = document.getElementById(`doc-${docId}`);
        if (el) {
            el.style.opacity = '0';
            el.style.transform = 'translateX(20px)';
            el.style.transition = '0.3s ease';
            setTimeout(() => {
                el.remove();
                loadDocuments();
            }, 300);
        }
    } catch (err) {
        showToast(err.message, 'error');
    }
}

// ── ServiceNow Lookup ────────────────────────────────────────────────

const snHostInput = document.getElementById('sn-host-input');

snHostInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') lookupHost();
});

async function lookupHost() {
    const host = snHostInput.value.trim();
    if (!host) {
        showToast('Enter a hostname.', 'error');
        return;
    }

    const resultDiv = document.getElementById('sn-result');
    const table = document.getElementById('sn-table');
    const btn = document.getElementById('btn-sn-lookup');

    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Looking up...';

    try {
        const resp = await fetch(`${API}/servicenow/host`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ host }),
        });

        if (!resp.ok) throw new Error('Lookup failed');

        const data = await resp.json();

        if (data.message) {
            // Error or not found
            table.innerHTML = `
        <tr>
          <td colspan="2" style="text-align:center;color:var(--warning);padding:24px;">
            ⚠️ ${escapeHTML(data.message)}
          </td>
        </tr>`;
        } else {
            table.innerHTML = `
        <tr><th>Name</th><td>${escapeHTML(data.name || '—')}</td></tr>
        <tr><th>IP Address</th><td>${escapeHTML(data.ip_address || '—')}</td></tr>
        <tr><th>OS</th><td>${escapeHTML(data.os || '—')}</td></tr>
        <tr><th>Location</th><td>${escapeHTML(data.location || '—')}</td></tr>
        <tr><th>Install Status</th><td>${escapeHTML(data.install_status || '—')}</td></tr>
      `;
        }

        resultDiv.classList.add('show');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg> Lookup`;
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// Initial load
loadDocuments();
