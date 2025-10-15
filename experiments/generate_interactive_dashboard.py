#!/usr/bin/env python3
"""
Generate interactive HTML dashboard for layer sweep results
"""

import json
import jsonlines
from pathlib import Path
from collections import defaultdict
import numpy as np

# Find all evaluation results
output_dir = Path("output/comparison")
results_by_model_layer = defaultdict(list)

print("Loading evaluation results...")
for jsonl_file in output_dir.rglob("probe_evaluate-v2.jsonl"):
    model = jsonl_file.parent.parent.name
    layer = jsonl_file.parent.name

    with jsonlines.open(jsonl_file) as reader:
        for item in reader:
            value = item["value"]
            results_by_model_layer[(model, layer)].append(value)

    print(f"  Loaded {model}/{layer}: {len(results_by_model_layer[(model, layer)])} evaluations")

print(f"\nTotal: {len(results_by_model_layer)} model/layer combinations")

# Extract unique datasets and group by collection
from repeng.datasets.elk.utils.collections import resolve_dataset_ids

all_datasets_set = set()
for results in results_by_model_layer.values():
    for r in results:
        all_datasets_set.add(r["train_dataset"])
        all_datasets_set.add(r["eval_dataset"])

# Group datasets by collection
collections = ["dlk", "repe", "got", "paper", "custom"]
dataset_groups = {}
for collection in collections:
    collection_datasets = resolve_dataset_ids(collection)
    for ds in collection_datasets:
        if ds in all_datasets_set:
            dataset_groups[ds] = collection

# Sort datasets by collection, then alphabetically within collection
all_datasets = []
group_boundaries = []
current_pos = 0

for collection in collections:
    collection_datasets = sorted([ds for ds in all_datasets_set if dataset_groups.get(ds) == collection])
    if collection_datasets:
        all_datasets.extend(collection_datasets)
        group_boundaries.append({
            "name": collection.upper(),
            "start": current_pos,
            "end": current_pos + len(collection_datasets)
        })
        current_pos += len(collection_datasets)

print(f"Datasets: {len(all_datasets)}")
print(f"Groups: {[f'{g['name']}({g['end']-g['start']})' for g in group_boundaries]}")

# Build data structure for JavaScript
data = {
    "datasets": all_datasets,
    "groups": group_boundaries,
    "models": {},
}

for (model, layer), results in results_by_model_layer.items():
    if model not in data["models"]:
        data["models"][model] = {}

    # Build matrices for this model/layer
    n = len(all_datasets)
    matrix_recovered = [[0.0 for _ in range(n)] for _ in range(n)]
    matrix_auc = [[0.0 for _ in range(n)] for _ in range(n)]
    matrix_accuracy = [[0.0 for _ in range(n)] for _ in range(n)]

    # Compute best in-distribution accuracies for recovered metric
    best_in_dist = {}
    for r in results:
        train_ds = r["train_dataset"]
        eval_ds = r["eval_dataset"]
        if train_ds == eval_ds:
            best_in_dist[train_ds] = r["accuracy"]

    for r in results:
        train_idx = all_datasets.index(r["train_dataset"])
        eval_idx = all_datasets.index(r["eval_dataset"])

        acc = r["accuracy"]
        auc = r["auc"]

        # Recovered accuracy
        eval_ds = r["eval_dataset"]
        if eval_ds in best_in_dist and best_in_dist[eval_ds] > 0:
            recovered = acc / best_in_dist[eval_ds]
        else:
            recovered = 0.0

        matrix_recovered[train_idx][eval_idx] = recovered
        matrix_auc[train_idx][eval_idx] = auc
        matrix_accuracy[train_idx][eval_idx] = acc

    data["models"][model][layer] = {
        "recovered": matrix_recovered,
        "auc": matrix_auc,
        "accuracy": matrix_accuracy,
    }

# Generate HTML
html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Layer Sweep Dashboard - Geometry of Truth</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0a0a0a;
            color: #e0e0e0;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
        }

        h1 {
            font-size: 28px;
            margin-bottom: 10px;
            color: #fff;
        }

        .subtitle {
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }

        .controls {
            background: #1a1a1a;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            border: 1px solid #333;
        }

        .control-row {
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .control-group {
            flex: 1;
            min-width: 200px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-size: 13px;
            font-weight: 600;
            color: #aaa;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        select, input[type="range"] {
            width: 100%;
            padding: 8px 12px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #fff;
            font-size: 14px;
        }

        select:focus {
            outline: none;
            border-color: #0066cc;
        }

        input[type="range"] {
            -webkit-appearance: none;
            height: 6px;
            background: #444;
            border-radius: 3px;
            padding: 0;
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            background: #0066cc;
            border-radius: 50%;
            cursor: pointer;
        }

        input[type="range"]::-moz-range-thumb {
            width: 18px;
            height: 18px;
            background: #0066cc;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }

        .range-value {
            display: inline-block;
            margin-left: 10px;
            color: #0066cc;
            font-weight: 600;
        }

        .toggle-buttons {
            display: flex;
            gap: 10px;
        }

        .toggle-btn {
            flex: 1;
            padding: 10px;
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 4px;
            color: #aaa;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 13px;
            font-weight: 600;
        }

        .toggle-btn:hover {
            background: #333;
        }

        .toggle-btn.active {
            background: #0066cc;
            border-color: #0066cc;
            color: #fff;
        }

        .viz-container {
            background: #1a1a1a;
            padding: 30px;
            border-radius: 8px;
            border: 1px solid #333;
        }

        .viz-header {
            margin-bottom: 20px;
        }

        .viz-title {
            font-size: 20px;
            font-weight: 600;
            color: #fff;
            margin-bottom: 5px;
        }

        .viz-subtitle {
            font-size: 13px;
            color: #888;
        }

        #heatmap {
            margin: 0 auto;
            display: block;
            max-width: 100%;
            height: auto;
        }

        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .stat-card {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #444;
        }

        .stat-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 5px;
        }

        .stat-value {
            font-size: 24px;
            font-weight: 700;
            color: #0066cc;
        }

        .legend {
            display: flex;
            align-items: center;
            justify-content: center;
            margin-top: 20px;
            gap: 10px;
            font-size: 12px;
        }

        .legend-gradient {
            width: 200px;
            height: 20px;
            border-radius: 3px;
            border: 1px solid #444;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Layer Sweep Dashboard</h1>
        <p class="subtitle">Interactive visualization of cross-dataset probe generalization across layers and models</p>

        <div class="controls">
            <div class="control-row">
                <div class="control-group">
                    <label>Model Family</label>
                    <div class="toggle-buttons">
                        <button class="toggle-btn active" data-family="qwen3" onclick="setFamily('qwen3')">Qwen3</button>
                        <button class="toggle-btn" data-family="llama2" onclick="setFamily('llama2')">LLaMA-2</button>
                    </div>
                </div>

                <div class="control-group">
                    <label>Model Size <span class="range-value" id="sizeValue">-</span></label>
                    <input type="range" id="sizeSlider" min="0" max="2" value="0" oninput="updateVisualization()">
                </div>

                <div class="control-group">
                    <label>Variant</label>
                    <div class="toggle-buttons">
                        <button class="toggle-btn active" data-variant="instruct" onclick="setVariant('instruct')">Instruct</button>
                        <button class="toggle-btn" data-variant="base" onclick="setVariant('base')">Base</button>
                    </div>
                </div>
            </div>

            <div class="control-row">
                <div class="control-group">
                    <label>Layer <span class="range-value" id="layerValue">-</span></label>
                    <input type="range" id="layerSlider" min="0" max="5" value="3" oninput="updateVisualization()">
                </div>

                <div class="control-group">
                    <label>Metric</label>
                    <select id="metricSelect" onchange="updateVisualization()">
                        <option value="accuracy">Accuracy / AUC (% correct)</option>
                        <option value="recovered">Recovered Accuracy (normalized)</option>
                    </select>
                </div>
            </div>
        </div>

        <div class="stats" id="stats">
            <div class="stat-card">
                <div class="stat-label">Avg In-Distribution</div>
                <div class="stat-value" id="statInDist">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Cross-Dataset</div>
                <div class="stat-value" id="statCrossDist">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Generalization Gap</div>
                <div class="stat-value" id="statGap">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Best Layer</div>
                <div class="stat-value" id="statBestLayer">-</div>
            </div>
        </div>

        <div class="viz-container">
            <div class="viz-header">
                <div class="viz-title" id="vizTitle">Cross-Dataset Generalization Matrix</div>
                <div class="viz-subtitle" id="vizSubtitle">Train dataset (rows) × Eval dataset (columns)</div>
            </div>

            <canvas id="heatmap" width="800" height="800"></canvas>

            <div class="legend">
                <span>0.0</span>
                <div class="legend-gradient" id="legendGradient"></div>
                <span>1.0</span>
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const DATA = """ + json.dumps(data, indent=2) + """;

        let currentFamily = 'qwen3';
        let currentVariant = 'instruct';
        let currentSize = null;

        // Initialize
        function init() {
            updateSizeOptions();
            updateVisualization();
        }

        function setFamily(family) {
            currentFamily = family;
            document.querySelectorAll('[data-family]').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.family === family);
            });
            updateSizeOptions();
            updateVisualization();
        }

        function setVariant(variant) {
            currentVariant = variant;
            document.querySelectorAll('[data-variant]').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.variant === variant);
            });
            updateVisualization();
        }

        function updateSizeOptions() {
            const slider = document.getElementById('sizeSlider');

            if (currentFamily === 'qwen3') {
                slider.max = 2;
                if (slider.value > 2) slider.value = 2;
            } else {
                slider.max = 1;
                if (slider.value > 1) slider.value = 1;
            }
        }

        function getModelName() {
            const sizeIdx = parseInt(document.getElementById('sizeSlider').value);

            if (currentFamily === 'qwen3') {
                const sizes = ['4b', '8b', '14b'];
                currentSize = sizes[sizeIdx];
                document.getElementById('sizeValue').textContent = currentSize.toUpperCase();
                return currentVariant === 'base' ? `qwen3-${currentSize}-base` : `qwen3-${currentSize}`;
            } else {
                const sizes = ['7b', '13b'];
                currentSize = sizes[sizeIdx];
                document.getElementById('sizeValue').textContent = currentSize.toUpperCase();
                return currentVariant === 'base' ? `llama2-${currentSize}` : `llama2-${currentSize}-chat`;
            }
        }

        function getAvailableLayers(modelName) {
            if (!DATA.models[modelName]) return [];
            return Object.keys(DATA.models[modelName]).sort((a, b) => {
                const numA = parseInt(a.replace('layer_h', ''));
                const numB = parseInt(b.replace('layer_h', ''));
                return numA - numB;
            });
        }

        function updateVisualization() {
            const modelName = getModelName();
            const layers = getAvailableLayers(modelName);

            if (layers.length === 0) {
                document.getElementById('vizTitle').textContent = `No data available for ${modelName}`;
                return;
            }

            const layerSlider = document.getElementById('layerSlider');
            layerSlider.max = layers.length - 1;

            const layerIdx = parseInt(layerSlider.value);
            const layer = layers[Math.min(layerIdx, layers.length - 1)];
            const layerNum = layer.replace('layer_h', '');

            document.getElementById('layerValue').textContent = `h${layerNum}`;

            const metric = document.getElementById('metricSelect').value;
            const matrix = DATA.models[modelName][layer][metric];

            // Update title
            const metricName = {
                'accuracy': 'Accuracy / AUC Score',
                'recovered': 'Recovered Accuracy (normalized by best in-dist)'
            }[metric];

            document.getElementById('vizTitle').textContent = `${modelName} - Layer h${layerNum} - ${metricName}`;

            // Draw heatmap
            drawHeatmap(matrix, DATA.datasets);

            // Calculate stats
            calculateStats(matrix, layers, modelName, metric);
        }

        function drawHeatmap(matrix, datasets) {
            const canvas = document.getElementById('heatmap');
            const ctx = canvas.getContext('2d');
            const n = datasets.length;

            const labelWidth = 140;
            const labelHeight = 140;
            const cellSize = 40;
            const matrixSize = cellSize * n;
            const groupLabelHeight = 20;

            canvas.width = labelWidth + matrixSize;
            canvas.height = labelHeight + matrixSize;

            // Clear
            ctx.fillStyle = '#0a0a0a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Find min/max for color scaling
            let min = Infinity, max = -Infinity;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const val = matrix[i][j];
                    if (val > max) max = val;
                    if (val < min) min = val;
                }
            }

            // Draw group labels at top
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            const groupColors = {
                'DLK': '#3b82f6',
                'REPE': '#8b5cf6',
                'GOT': '#ec4899',
                'PAPER': '#f59e0b',
                'CUSTOM': '#10b981'
            };

            for (const group of DATA.groups) {
                const startX = labelWidth + group.start * cellSize;
                const width = (group.end - group.start) * cellSize;
                const x = startX + width / 2;
                const y = labelHeight - 115;

                ctx.fillStyle = groupColors[group.name] || '#666';
                ctx.fillText(group.name, x, y);
            }

            // Draw column labels (top, rotated)
            ctx.save();
            ctx.fillStyle = '#888';
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';

            for (let j = 0; j < n; j++) {
                const x = labelWidth + j * cellSize + cellSize / 2;
                const y = labelHeight - 5;

                ctx.save();
                ctx.translate(x, y);
                ctx.rotate(-Math.PI / 4);

                const label = datasets[j].length > 15 ? datasets[j].substring(0, 13) + '..' : datasets[j];
                ctx.fillText(label, 0, 0);
                ctx.restore();
            }
            ctx.restore();

            // Draw group labels on left
            ctx.font = 'bold 11px sans-serif';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';

            for (const group of DATA.groups) {
                const startY = labelHeight + group.start * cellSize;
                const height = (group.end - group.start) * cellSize;
                const y = startY + height / 2;
                const x = 35;

                ctx.fillStyle = groupColors[group.name] || '#666';
                ctx.fillText(group.name, x, y);
            }

            // Draw row labels (left)
            ctx.fillStyle = '#888';
            ctx.font = '10px monospace';
            ctx.textAlign = 'right';
            ctx.textBaseline = 'middle';

            for (let i = 0; i < n; i++) {
                const x = labelWidth - 5;
                const y = labelHeight + i * cellSize + cellSize / 2;

                const label = datasets[i].length > 15 ? datasets[i].substring(0, 13) + '..' : datasets[i];
                ctx.fillText(label, x, y);
            }

            // Draw cells with values
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.font = 'bold 9px monospace';

            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    const val = matrix[i][j];
                    const normalized = (val - min) / (max - min);

                    // Color scale: blue (low) -> cyan -> green -> yellow -> red (high)
                    const color = getColor(normalized);

                    const x = labelWidth + j * cellSize;
                    const y = labelHeight + i * cellSize;

                    ctx.fillStyle = color;
                    ctx.fillRect(x, y, cellSize - 1, cellSize - 1);

                    // Draw percentage value
                    const percent = (val * 100).toFixed(0);

                    // Use black text for bright cells (yellow/green), white for dark cells (red/orange)
                    const brightness = normalized;
                    ctx.fillStyle = brightness > 0.45 ? '#000' : '#fff';
                    ctx.fillText(percent, x + cellSize / 2, y + cellSize / 2);
                }
            }

            // Draw group separators (thicker lines between groups)
            ctx.strokeStyle = '#666';
            ctx.lineWidth = 2;

            for (const group of DATA.groups) {
                if (group.end < n) {
                    // Vertical separator
                    const x = labelWidth + group.end * cellSize;
                    ctx.beginPath();
                    ctx.moveTo(x, labelHeight);
                    ctx.lineTo(x, labelHeight + matrixSize);
                    ctx.stroke();

                    // Horizontal separator
                    const y = labelHeight + group.end * cellSize;
                    ctx.beginPath();
                    ctx.moveTo(labelWidth, y);
                    ctx.lineTo(labelWidth + matrixSize, y);
                    ctx.stroke();
                }
            }

            // Update legend gradient
            const legendGradient = document.getElementById('legendGradient');
            legendGradient.style.background = 'linear-gradient(to right, #dc2626, #f97316, #facc15, #84cc16, #22c55e)';
        }

        function getColor(normalized) {
            // 5-color gradient: red (bad) -> orange -> yellow -> lime -> green (good)
            // Reversed so green = high = good, red = low = bad
            if (normalized < 0.25) {
                const t = normalized / 0.25;
                return interpolateColor('#dc2626', '#f97316', t);  // red -> orange
            } else if (normalized < 0.5) {
                const t = (normalized - 0.25) / 0.25;
                return interpolateColor('#f97316', '#facc15', t);  // orange -> yellow
            } else if (normalized < 0.75) {
                const t = (normalized - 0.5) / 0.25;
                return interpolateColor('#facc15', '#84cc16', t);  // yellow -> lime
            } else {
                const t = (normalized - 0.75) / 0.25;
                return interpolateColor('#84cc16', '#22c55e', t);  // lime -> green
            }
        }

        function interpolateColor(color1, color2, t) {
            const c1 = parseInt(color1.slice(1), 16);
            const c2 = parseInt(color2.slice(1), 16);

            const r1 = (c1 >> 16) & 0xff;
            const g1 = (c1 >> 8) & 0xff;
            const b1 = c1 & 0xff;

            const r2 = (c2 >> 16) & 0xff;
            const g2 = (c2 >> 8) & 0xff;
            const b2 = c2 & 0xff;

            const r = Math.round(r1 + (r2 - r1) * t);
            const g = Math.round(g1 + (g2 - g1) * t);
            const b = Math.round(b1 + (b2 - b1) * t);

            return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, '0')}`;
        }

        function calculateStats(matrix, layers, modelName, metric) {
            const n = matrix.length;

            // In-distribution (diagonal)
            let inDistSum = 0, inDistCount = 0;
            for (let i = 0; i < n; i++) {
                if (matrix[i][i] > 0) {
                    inDistSum += matrix[i][i];
                    inDistCount++;
                }
            }
            const avgInDist = inDistCount > 0 ? inDistSum / inDistCount : 0;

            // Cross-dataset (off-diagonal)
            let crossDistSum = 0, crossDistCount = 0;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (i !== j && matrix[i][j] > 0) {
                        crossDistSum += matrix[i][j];
                        crossDistCount++;
                    }
                }
            }
            const avgCrossDist = crossDistCount > 0 ? crossDistSum / crossDistCount : 0;

            // Find best layer
            let bestLayer = '-';
            let bestScore = -Infinity;
            for (const layer of layers) {
                const mat = DATA.models[modelName][layer][metric];
                let score = 0, count = 0;
                for (let i = 0; i < n; i++) {
                    for (let j = 0; j < n; j++) {
                        if (mat[i][j] > 0) {
                            score += mat[i][j];
                            count++;
                        }
                    }
                }
                const avgScore = count > 0 ? score / count : 0;
                if (avgScore > bestScore) {
                    bestScore = avgScore;
                    bestLayer = layer.replace('layer_h', 'h');
                }
            }

            // Update display
            // Note: For 'recovered' metric, in-dist is always 1.0 by definition
            if (metric === 'recovered') {
                document.getElementById('statInDist').textContent = '1.000*';
            } else {
                document.getElementById('statInDist').textContent = avgInDist.toFixed(3);
            }
            document.getElementById('statCrossDist').textContent = avgCrossDist.toFixed(3);
            document.getElementById('statGap').textContent = (avgInDist - avgCrossDist).toFixed(3);
            document.getElementById('statBestLayer').textContent = bestLayer;
        }

        // Initialize on load
        init();
    </script>
</body>
</html>
"""

# Write HTML file
output_file = Path("output/dashboard.html")
output_file.write_text(html_content)

print(f"\n✓ Dashboard generated: {output_file}")
print(f"  Open in browser: file://{output_file.absolute()}")
