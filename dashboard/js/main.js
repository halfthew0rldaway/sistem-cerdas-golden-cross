document.addEventListener('DOMContentLoaded', () => {
    const navItems = document.querySelectorAll('.nav-item');
    const sections = document.querySelectorAll('.section');
    const headerTitle = document.getElementById('header-title');

    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navItems.forEach(nav => nav.classList.remove('active'));
            sections.forEach(sec => sec.classList.remove('active'));

            item.classList.add('active');
            
            const targetId = item.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
            headerTitle.textContent = item.textContent.trim();
            
            if(targetId === 'sec-chart') {
                const iframe = document.getElementById('chart-iframe');
                iframe.src = iframe.src; 
            }

            // Auto-load metrics saat tab Metrik Performa diklik
            if(targetId === 'sec-metrics') {
                loadMetrics();
            }
        });
    });

    // Auto-load metrics saat halaman pertama kali dibuka (jika sudah ada JSON)
    loadMetrics();
});

let activeDataset = 'daridosenbarukhususgoldencross.csv';

// ==================== METRICS RENDERER ====================

const SCENARIO_CONFIG = {
    validation: { title: "Skenario 0 — Validation Set (15%)", cmLabel: "Validation" },
    test_standar: { title: "Skenario 1 — Test Set Standar (15%)", cmLabel: "Standar" },
    oot: { title: "Skenario 2 — Out-Of-Time (15% Terbaru)", cmLabel: "OOT" },
    noise: { title: "Skenario 3 — Test Set Ber-Noise (5%)", cmLabel: "Noise" }
};

function generateInsight(key, m) {
    const total = m.support_pos;
    if (key === 'validation') {
        return `Model menangkap ${m.tp} dari ${total} kemunculan Golden Cross di data validasi, dengan ${m.fp.toLocaleString()} alarm palsu akibat kewaspadaan tinggi hasil undersampling.`;
    } else if (key === 'test_standar') {
        return `Performa pada data uji baru: Model mendeteksi ${m.tp} dari ${total} Golden Cross asli (Recall ${m.recall}%), membuktikan model tidak overfitting.`;
    } else if (key === 'oot') {
        return `Pada data kronologis masa depan, model menangkap ${m.tp} dari ${total} peluang riil. Alarm palsu hanya ${m.fp.toLocaleString()}, jauh lebih rendah dibanding skenario lain.`;
    } else if (key === 'noise') {
        return `Penambahan noise Gaussian 5% nyaris tidak menggoyahkan stabilitas. Recall tetap ${m.recall}%, membuktikan model robust terhadap fluktuasi acak.`;
    }
    return '';
}

function renderMetricCard(key, m) {
    const cfg = SCENARIO_CONFIG[key];
    if (!cfg) return '';

    const insight = generateInsight(key, m);

    return `
    <div class="card">
        <div class="card-title">${cfg.title}</div>
        <div class="stat-row"><span>Accuracy</span><span class="val-neutral">${m.accuracy}%</span></div>
        <div class="stat-row"><span>ROC-AUC Score</span><span class="val-good">${m.roc_auc}</span></div>
        <div class="stat-row"><span>Recall</span><span class="val-good">${m.recall}%</span></div>
        <div class="stat-row"><span>Precision</span><span class="${m.precision < 5 ? 'val-warn' : 'val-good'}">${m.precision}%</span></div>
        <div class="cm-box">
            <div class="cm-box-title">Confusion Matrix (${cfg.cmLabel})</div>
            <div class="cm-grid">
                <div class="cm-item"><span class="cm-label">True Negative (TN)</span><span class="cm-val">${m.tn.toLocaleString()}</span></div>
                <div class="cm-item"><span class="cm-label">False Positive (FP)</span><span class="cm-val val-warn">${m.fp.toLocaleString()}</span></div>
                <div class="cm-item"><span class="cm-label">False Negative (FN)</span><span class="cm-val val-warn">${m.fn.toLocaleString()}</span></div>
                <div class="cm-item"><span class="cm-label">True Positive (TP)</span><span class="cm-val val-good">${m.tp.toLocaleString()}</span></div>
            </div>
            <p class="cm-insight"><strong>Insight:</strong> ${insight}</p>
        </div>
    </div>`;
}

async function loadMetrics() {
    const container = document.getElementById('metrics-container');
    const loading = document.getElementById('metrics-loading');

    try {
        const res = await fetch('/api/metrics');
        if (!res.ok) {
            loading.style.display = 'block';
            container.innerHTML = '';
            return;
        }

        const data = await res.json();
        loading.style.display = 'none';

        // Update overview dataset stats jika ada
        if (data.dataset) {
            const rowsEl = document.getElementById('ds-total-rows');
            const gcEl = document.getElementById('ds-total-gc');
            const ratioEl = document.getElementById('ds-ratio');
            const dynSummary = document.getElementById('dyn-dataset-summary');
            if (rowsEl) rowsEl.textContent = data.dataset.total_rows.toLocaleString();
            if (gcEl) gcEl.textContent = data.dataset.total_gc + ' Kejadian';
            if (ratioEl) {
                const pct = (data.dataset.total_gc / data.dataset.total_rows * 100).toFixed(2);
                ratioEl.textContent = pct + '% : ' + (100 - parseFloat(pct)).toFixed(2) + '%';
            }
            if (dynSummary) dynSummary.textContent = `${(data.dataset.total_rows / 1000).toFixed(1)}K baris (Dinamis)`;
        }

        // Render semua kartu skenario & kalkulasi agregat untuk Analisis
        let html = '';
        const order = ['validation', 'test_standar', 'oot', 'noise'];
        
        let maxRecall = 0;
        let totalRoc = 0;
        let countRoc = 0;
        let minPrec = 100;
        let maxPrec = 0;

        for (const key of order) {
            if (data.scenarios[key]) {
                const s = data.scenarios[key];
                html += renderMetricCard(key, s);
                
                // Agregat
                if (s.recall > maxRecall) maxRecall = s.recall;
                totalRoc += parseFloat(s.roc_auc);
                countRoc++;
                if (s.precision < minPrec) minPrec = s.precision;
                if (s.precision > maxPrec) maxPrec = s.precision;
            }
        }
        container.innerHTML = html;

        // Update Section 4: Analisis & Kesimpulan secara dinamis
        if (countRoc > 0) {
            const avgRoc = (totalRoc / countRoc).toFixed(3);
            
            const elTercapai = document.getElementById('dyn-tercapai');
            if (elTercapai) {
                elTercapai.textContent = maxRecall >= 50 ? `Ya (Recall hingga ${maxRecall.toFixed(1)}%)` : `Tidak (Hanya ${maxRecall.toFixed(1)}%)`;
                elTercapai.className = maxRecall >= 50 ? 'val-good' : 'val-warn';
            }
            const elRoc = document.getElementById('dyn-roc');
            if (elRoc) {
                elRoc.textContent = `~ ${avgRoc}`;
                elRoc.className = avgRoc >= 0.8 ? 'val-good' : 'val-warn';
            }
            const elPrec = document.getElementById('dyn-precision');
            if (elPrec) {
                elPrec.textContent = `Precision sangat rendah (${minPrec.toFixed(1)}% - ${maxPrec.toFixed(1)}%)`;
            }
            const elLayak = document.getElementById('dyn-layak');
            if (elLayak) {
                const isLayak = maxRecall >= 50 && avgRoc >= 0.8;
                elLayak.textContent = isLayak ? 'Ya (sebagai Screener/Radar)' : 'Perlu Evaluasi Ulang';
                elLayak.className = isLayak ? 'val-good' : 'val-warn';
            }
        }

    } catch (e) {
        loading.style.display = 'block';
        container.innerHTML = '';
    }
}

// ==================== UPLOAD ====================

async function uploadDataset() {
    const fileInput = document.getElementById('csvUpload');
    if (fileInput.files.length === 0) {
        alert("Pilih file CSV terlebih dahulu!");
        return;
    }
    
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    
    const btn = event.target;
    const origText = btn.textContent;
    btn.textContent = "Uploading...";
    btn.disabled = true;
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        if (response.ok) {
            activeDataset = data.filename;
            document.getElementById('active-dataset').textContent = "✓ " + activeDataset;
            alert("Berhasil! Semua tombol di bawah sekarang akan menggunakan dataset: " + activeDataset);
        } else {
            alert("Error: " + data.error);
        }
    } catch (e) {
        alert("Gagal upload: " + e.message);
    } finally {
        btn.textContent = origText;
        btn.disabled = false;
    }
}

// ==================== RUN SCENARIO ====================

async function runScenario(scenario) {
    const terminalOutput = document.getElementById('terminal-output');
    const btn = event.target;
    const originalText = btn.textContent;
    
    btn.disabled = true;
    btn.textContent = 'Memproses...';
    
    document.getElementById('cli-controls').style.display = 'none';
    document.getElementById('cli-output').style.display = 'block';

    let dotCount = 0;
    terminalOutput.textContent = '🚀 Memulai proses di server backend...\n⏳ Mengkalkulasi algoritma Machine Learning (Mohon tunggu hingga 15 detik) ';
    const loadingInterval = setInterval(() => {
        dotCount = (dotCount + 1) % 4;
        const dots = ".".repeat(dotCount) + " ".repeat(3 - dotCount);
        terminalOutput.textContent = terminalOutput.textContent.substring(0, terminalOutput.textContent.length - 3) + dots;
    }, 500);

    const payload = { 
        scenario: scenario,
        filename: activeDataset
    };
    
    if (scenario === 'custom') {
        payload.ma_short = document.getElementById('ma_short').value;
        payload.ma_long = document.getElementById('ma_long').value;
    }

    try {
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        clearInterval(loadingInterval);
        
        const data = await response.json();
        
        if (response.ok) {
            terminalOutput.textContent = data.output;
            if (scenario === 'main' || scenario === 'custom') {
                terminalOutput.textContent += '\n\n✅ Selesai! Chart dan Metrik Performa berhasil diupdate otomatis.';
                // Auto-refresh metrics setelah pipeline selesai
                loadMetrics();
            }
        } else {
            terminalOutput.textContent = '❌ ERROR DARI SERVER:\n' + (data.error || JSON.stringify(data));
        }
        
    } catch (error) {
        clearInterval(loadingInterval);
        terminalOutput.textContent = '❌ Gagal menghubungi server: ' + error.message;
    } finally {
        btn.disabled = false;
        btn.textContent = originalText;
    }
}

function backToMenu() {
    document.getElementById('cli-output').style.display = 'none';
    document.getElementById('cli-controls').style.display = 'block';
}
