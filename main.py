import asyncio
from typing import Dict, Any, List
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn
import json

# Importa o motor científico do seu script original
# Certifique-se de que seu script se chama ninja_engine.py
import ninja_engine as engine

app = FastAPI(title="NINJA SUPREME 2.0 API")

# --- Estado Global em Memória (Simples para demonstração) ---
# Em produção, usaria Redis ou Banco de Dados
job_store: Dict[str, Any] = {
    "status": "IDLE",  # IDLE, RUNNING, COMPLETED, ERROR
    "data": None,
    "logs": []
}

# --- Lógica de Execução em Background ---
def run_scientific_calculation():
    """
    Executa a lógica pesada do NINJA SUPREME.
    Forçamos o modo SANITY_CHECK para a web não demorar horas.
    """
    job_store["status"] = "RUNNING"
    job_store["logs"] = ["Inicializando NINJA SUPREME 2.0...", "Carregando configurações (SANITY_CHECK)..."]

    try:
        # 1. Configuração (Forçando modo rápido para demonstração web)
        # Se quiser precisão total, mude para "PAPER_READY", mas vai demorar muito.
        cfg = engine.NinjaConfig(mode="SANITY_CHECK")

        job_store["logs"].append("Carregando datasets cosmológicos...")
        data = engine.NinjaDataVectorized(cfg)

        # 2. Rodar Modelo LCDM
        job_store["logs"].append("Executando Nested Sampling: ΛCDM...")
        res_lcdm = engine.run_nested_model(
            "LCDM",
            loglike=engine.loglike_lcdm,
            prior_transform=engine.prior_transform_LCDM,
            data=data,
            cfg=cfg,
            n_dim=6,
        )

        # 3. Rodar Modelo DUT (Dead Universe Theory)
        job_store["logs"].append("Executando Nested Sampling: DUT (ExtractoDAO)...")
        res_dut = engine.run_nested_model(
            "DUT",
            loglike=engine.loglike_dut,
            prior_transform=engine.prior_transform_DUT,
            data=data,
            cfg=cfg,
            n_dim=6,
        )

        # 4. Calcular Bayes Factor
        logB = res_dut["logZ"] - res_lcdm["logZ"]
        job_store["logs"].append(f"Cálculo finalizado. Log Bayes Factor: {logB:.3f}")

        # 5. Preparar dados para o Frontend
        # Precisamos converter numpy arrays para listas para o JSON
        idx_max_lcdm = np.argmax(res_lcdm["weights"])
        best_lcdm = res_lcdm["samples"][idx_max_lcdm]

        idx_max_dut = np.argmax(res_dut["weights"])
        best_dut = res_dut["samples"][idx_max_dut]

        final_results = {
            "log_bayes_factor": float(logB),
            "evidence_lcdm": float(res_lcdm["logZ"]),
            "evidence_dut": float(res_dut["logZ"]),
            "best_fit": {
                "lcdm": {
                    "H0": float(best_lcdm[0]), "Om": float(best_lcdm[1]), "S8": float(best_lcdm[2])
                },
                "dut": {
                    "H0": float(best_dut[0]), "Om": float(best_dut[1]), "S8": float(best_dut[2]),
                    "w0": float(best_dut[3]), "wa": float(best_dut[4]), "xi": float(best_dut[5])
                }
            },
            # Samples simplificados para plotar (apenas os últimos 200 para não pesar o JSON)
            "samples_dut_h0": res_dut["samples"][-200:, 0].tolist(),
            "samples_dut_om": res_dut["samples"][-200:, 1].tolist()
        }

        job_store["data"] = final_results
        job_store["status"] = "COMPLETED"

    except Exception as e:
        job_store["status"] = "ERROR"
        job_store["logs"].append(f"ERRO CRÍTICO: {str(e)}")
        print(e)

# --- Endpoints da API ---

@app.post("/api/run")
async def start_run(background_tasks: BackgroundTasks):
    if job_store["status"] == "RUNNING":
        return {"message": "Job already running"}

    # Limpa estado anterior
    job_store["data"] = None
    job_store["logs"] = []

    # Inicia tarefa em background
    background_tasks.add_task(run_scientific_calculation)
    return {"message": "Simulation started", "status": "RUNNING"}

@app.get("/api/status")
async def get_status():
    return job_store

# --- Frontend HTML ---

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NINJA SUPREME 2.0 | ExtractoDAO Interface</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Courier New', Courier, monospace; }
        .neon-text { text-shadow: 0 0 10px #3b82f6; }
        .glass-panel { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid #334155; }
    </style>
</head>
<body class="p-8">

    <div class="max-w-6xl mx-auto">
        <header class="mb-10 text-center border-b border-gray-700 pb-6">
            <h1 class="text-4xl font-bold text-blue-400 neon-text mb-2">NINJA SUPREME 2.0</h1>
            <p class="text-gray-400 text-sm">Unified Bayesian Cosmology Engine (ΛCDM vs DUT)</p>
            <p class="text-xs text-gray-500 mt-1">© 2025 ExtractoDAO Labs</p>
        </header>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            <div class="glass-panel p-6 rounded-lg md:col-span-1">
                <h2 class="text-xl font-bold mb-4 text-green-400">Control Center</h2>
                <p class="text-sm text-gray-400 mb-4">Mode: <span class="text-yellow-400">WEB DEMO (Low Precision)</span></p>
                <button id="btnStart" onclick="startSimulation()" class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-4 rounded transition shadow-[0_0_15px_rgba(37,99,235,0.5)]">
                    INITIALIZE ENGINE
                </button>
                <div id="statusIndicator" class="mt-4 text-center text-sm font-bold text-gray-500">SYSTEM IDLE</div>
            </div>

            <div class="glass-panel p-6 rounded-lg md:col-span-2">
                <h2 class="text-xl font-bold mb-2 text-blue-300">System Logs</h2>
                <div id="consoleOutput" class="h-48 overflow-y-auto bg-black p-4 rounded text-xs text-green-500 font-mono border border-gray-700">
                    > Ready to initialize...
                </div>
            </div>
        </div>

        <div id="resultsSection" class="hidden animate-fade-in">

            <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div class="glass-panel p-6 rounded-lg text-center">
                    <h3 class="text-gray-400 text-sm">Log Bayes Factor (DUT/ΛCDM)</h3>
                    <div id="valLogB" class="text-4xl font-bold text-white mt-2">--</div>
                    <div id="verdict" class="text-xs mt-2 uppercase tracking-widest">--</div>
                </div>
                <div class="glass-panel p-6 rounded-lg text-center">
                    <h3 class="text-gray-400 text-sm">Evidence ΛCDM (logZ)</h3>
                    <div id="valZLCDM" class="text-2xl font-bold text-gray-300 mt-2">--</div>
                </div>
                <div class="glass-panel p-6 rounded-lg text-center">
                    <h3 class="text-gray-400 text-sm">Evidence DUT (logZ)</h3>
                    <div id="valZDUT" class="text-2xl font-bold text-purple-300 mt-2">--</div>
                </div>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass-panel p-6 rounded-lg">
                    <h3 class="text-xl font-bold mb-4 border-b border-gray-600 pb-2">Best Fit Parameters</h3>
                    <table class="w-full text-sm text-left">
                        <thead>
                            <tr class="text-gray-500"><th class="pb-2">Param</th><th class="pb-2">ΛCDM</th><th class="pb-2 text-purple-400">DUT</th></tr>
                        </thead>
                        <tbody class="divide-y divide-gray-800">
                            <tr><td class="py-2">H0</td><td id="tbl_lcdm_h0" class="text-gray-300"></td><td id="tbl_dut_h0" class="text-purple-300 font-bold"></td></tr>
                            <tr><td class="py-2">Ωm</td><td id="tbl_lcdm_om" class="text-gray-300"></td><td id="tbl_dut_om" class="text-purple-300 font-bold"></td></tr>
                            <tr><td class="py-2">S8</td><td id="tbl_lcdm_s8" class="text-gray-300"></td><td id="tbl_dut_s8" class="text-purple-300 font-bold"></td></tr>
                            <tr><td class="py-2">w0</td><td class="text-gray-600">-1.0 (fixed)</td><td id="tbl_dut_w0" class="text-purple-300 font-bold"></td></tr>
                            <tr><td class="py-2">wa</td><td class="text-gray-600">0.0 (fixed)</td><td id="tbl_dut_wa" class="text-purple-300 font-bold"></td></tr>
                            <tr><td class="py-2">ξ (Interaction)</td><td class="text-gray-600">0.0 (fixed)</td><td id="tbl_dut_xi" class="text-purple-300 font-bold"></td></tr>
                        </tbody>
                    </table>
                </div>

                <div class="glass-panel p-6 rounded-lg">
                    <h3 class="text-xl font-bold mb-4">Posterior Distribution (H0 vs Ωm)</h3>
                    <canvas id="chartPosterior"></canvas>
                </div>
            </div>

            <div class="mt-8 text-center">
                <div class="p-4 bg-yellow-900/30 border border-yellow-700 rounded text-yellow-200 text-xs">
                    WARNING: This web interface used "SANITY_CHECK" mode (low precision) for demonstration purposes.
                    Scientific results require "PAPER_READY" mode executed via CLI.
                </div>
            </div>
        </div>
    </div>

    <script>
        let pollingInterval = null;

        async function startSimulation() {
            const btn = document.getElementById('btnStart');
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');
            btn.innerText = "PROCESSING...";

            document.getElementById('resultsSection').classList.add('hidden');

            try {
                const res = await fetch('/api/run', { method: 'POST' });
                const data = await res.json();

                if(data.status === 'RUNNING') {
                    startPolling();
                }
            } catch (err) {
                console.error(err);
                alert("Error connecting to API");
                resetUI();
            }
        }

        function startPolling() {
            if (pollingInterval) clearInterval(pollingInterval);
            pollingInterval = setInterval(checkStatus, 2000);
        }

        async function checkStatus() {
            const res = await fetch('/api/status');
            const statusData = await res.json();

            // Update Logs
            const consoleDiv = document.getElementById('consoleOutput');
            consoleDiv.innerHTML = statusData.logs.map(l => `<div>> ${l}</div>`).join('');
            consoleDiv.scrollTop = consoleDiv.scrollHeight;

            // Update Status Indicator
            const statusInd = document.getElementById('statusIndicator');
            statusInd.innerText = "STATUS: " + statusData.status;

            if (statusData.status === 'COMPLETED') {
                clearInterval(pollingInterval);
                renderResults(statusData.data);
                resetUI();
            } else if (statusData.status === 'ERROR') {
                clearInterval(pollingInterval);
                resetUI();
                alert("Simulation Failed. Check logs.");
            }
        }

        function renderResults(data) {
            document.getElementById('resultsSection').classList.remove('hidden');

            // Metrics
            document.getElementById('valLogB').innerText = data.log_bayes_factor.toFixed(2);
            document.getElementById('valZLCDM').innerText = data.evidence_lcdm.toFixed(2);
            document.getElementById('valZDUT').innerText = data.evidence_dut.toFixed(2);

            const verdict = document.getElementById('verdict');
            if (data.log_bayes_factor > 1) {
                verdict.innerText = "STRONG EVIDENCE FOR DUT";
                verdict.className = "text-xs mt-2 uppercase tracking-widest text-green-400";
            } else if (data.log_bayes_factor < -1) {
                verdict.innerText = "STRONG EVIDENCE FOR ΛCDM";
                verdict.className = "text-xs mt-2 uppercase tracking-widest text-blue-400";
            } else {
                verdict.innerText = "INCONCLUSIVE";
                verdict.className = "text-xs mt-2 uppercase tracking-widest text-gray-400";
            }

            // Table
            document.getElementById('tbl_lcdm_h0').innerText = data.best_fit.lcdm.H0.toFixed(2);
            document.getElementById('tbl_lcdm_om').innerText = data.best_fit.lcdm.Om.toFixed(3);
            document.getElementById('tbl_lcdm_s8').innerText = data.best_fit.lcdm.S8.toFixed(3);

            document.getElementById('tbl_dut_h0').innerText = data.best_fit.dut.H0.toFixed(2);
            document.getElementById('tbl_dut_om').innerText = data.best_fit.dut.Om.toFixed(3);
            document.getElementById('tbl_dut_s8').innerText = data.best_fit.dut.S8.toFixed(3);
            document.getElementById('tbl_dut_w0').innerText = data.best_fit.dut.w0.toFixed(3);
            document.getElementById('tbl_dut_wa').innerText = data.best_fit.dut.wa.toFixed(3);
            document.getElementById('tbl_dut_xi').innerText = data.best_fit.dut.xi.toFixed(4);

            // Chart
            const ctx = document.getElementById('chartPosterior').getContext('2d');

            // Create scatter data from samples
            const scatterData = data.samples_dut_h0.map((h0, i) => ({
                x: data.samples_dut_om[i],
                y: h0
            }));

            new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'DUT Posterior Samples',
                        data: scatterData,
                        backgroundColor: 'rgba(168, 85, 247, 0.6)'
                    }]
                },
                options: {
                    scales: {
                        x: { title: { display: true, text: 'Omega Matter (Ωm)', color: '#9ca3af' }, grid: { color: '#374151' } },
                        y: { title: { display: true, text: 'Hubble Constant (H0)', color: '#9ca3af' }, grid: { color: '#374151' } }
                    },
                    plugins: { legend: { labels: { color: '#e5e7eb' } } }
                }
            });
        }

        function resetUI() {
            const btn = document.getElementById('btnStart');
            btn.disabled = false;
            btn.classList.remove('opacity-50', 'cursor-not-allowed');
            btn.innerText = "RE-INITIALIZE ENGINE";
        }
    </script>
</body>
</html>
    """

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)