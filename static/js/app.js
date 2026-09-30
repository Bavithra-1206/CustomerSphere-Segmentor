/**
 * CustomerIQ - Interactive Dashboard Controller
 */

// Color palette for clusters
const CLUSTER_COLORS = [
    '#8b5cf6', // Cluster 0: Electric Purple
    '#06b6d4', // Cluster 1: Cyan
    '#ec4899', // Cluster 2: Pink
    '#10b981', // Cluster 3: Emerald
    '#f59e0b', // Cluster 4: Amber
    '#3b82f6', // Cluster 5: Blue
    '#a855f7', // Cluster 6: Violet
    '#14b8a6', // Cluster 7: Teal
    '#f43f5e', // Cluster 8: Rose
    '#eab308'  // Cluster 9: Yellow
];

// App State
const state = {
    dataset: 'data/mall_customers.csv',
    scaling: 'standard',
    selectedFeatures: [],
    numericFeatures: [],
    k: 5,
    clusterData: null,
    columnInfo: null,
    currentVizMode: '2D',
    tableData: []
};

// Chart instances
let chartElbowInstance = null;
let chartSilhouetteInstance = null;
let chartMeansInstance = null;

document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    loadDataset(state.dataset);
});

function initEventListeners() {
    // Dataset Selector
    document.getElementById('datasetSelect').addEventListener('change', (e) => {
        state.dataset = e.target.value;
        loadDataset(state.dataset);
    });

    // Scaling Radios
    document.querySelectorAll('input[name="scaling"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.scaling = e.target.value;
            document.querySelectorAll('.radio-btn').forEach(btn => btn.classList.remove('active'));
            if (e.target.value === 'standard') {
                document.getElementById('lblStandard').classList.add('active');
            } else {
                document.getElementById('lblMinMax').classList.add('active');
            }
        });
    });

    // K-Slider
    const kSlider = document.getElementById('kSlider');
    kSlider.addEventListener('input', (e) => {
        state.k = parseInt(e.target.value);
        document.getElementById('kValBadge').innerText = state.k;
    });

    // Run Clustering Button
    document.getElementById('btnRunClustering').addEventListener('click', runClustering);

    // Evaluate Optimal K Button
    document.getElementById('btnRunOptimalK').addEventListener('click', () => {
        switchTab('tabElbowSilhouette');
        runOptimalKEvaluation();
    });

    // Tabs Navigation
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const targetTab = e.currentTarget.getAttribute('data-tab');
            switchTab(targetTab);
        });
    });

    // Viz 2D / 3D Toggles
    document.getElementById('btnViz2D').addEventListener('click', () => {
        state.currentVizMode = '2D';
        document.getElementById('btnViz2D').classList.add('active');
        document.getElementById('btnViz3D').classList.remove('active');
        renderPCAPlot();
    });
    document.getElementById('btnViz3D').addEventListener('click', () => {
        state.currentVizMode = '3D';
        document.getElementById('btnViz3D').classList.add('active');
        document.getElementById('btnViz2D').classList.remove('active');
        renderPCAPlot();
    });

    // Predictor Form
    document.getElementById('predictorForm').addEventListener('submit', handlePrediction);

    // Table Search
    document.getElementById('tableSearchInput').addEventListener('input', (e) => {
        filterTable(e.target.value);
    });

    // Export Data Button
    document.getElementById('btnExportData').addEventListener('click', exportCSV);
}

function switchTab(tabId) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    if (activeBtn) activeBtn.classList.add('active');

    const activePane = document.getElementById(tabId);
    if (activePane) activePane.classList.add('active');

    // Resize Plotly if returning to viz tab
    if (tabId === 'tabClusterViz') {
        setTimeout(() => Plotly.Plots.resize('plotlyClusterContainer'), 100);
    }
}

// --------------------------------------------------------------------------
// API Calls & Data Loading
// --------------------------------------------------------------------------

async function loadDataset(filename) {
    try {
        const res = await fetch('/api/load_data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename })
        });
        const data = await res.json();

        if (data.status === 'success') {
            state.columnInfo = data.info;
            state.numericFeatures = data.info.numeric_features;
            state.selectedFeatures = [...data.info.numeric_features];

            document.getElementById('infoRows').innerText = data.info.total_rows;
            document.getElementById('infoFeatCount').innerText = state.selectedFeatures.length;

            renderFeatureCheckboxes();
            renderPredictorFields();
            
            // Trigger initial clustering automatically
            runClustering();
        }
    } catch (err) {
        console.error('Error loading dataset:', err);
    }
}

function renderFeatureCheckboxes() {
    const container = document.getElementById('featureCheckboxes');
    container.innerHTML = '';

    state.numericFeatures.forEach(feat => {
        const label = document.createElement('label');
        label.className = 'checkbox-item';
        
        const isChecked = state.selectedFeatures.includes(feat);
        label.innerHTML = `
            <input type="checkbox" value="${feat}" ${isChecked ? 'checked' : ''}>
            <span>${feat}</span>
        `;

        label.querySelector('input').addEventListener('change', (e) => {
            if (e.target.checked) {
                if (!state.selectedFeatures.includes(feat)) state.selectedFeatures.push(feat);
            } else {
                state.selectedFeatures = state.selectedFeatures.filter(f => f !== feat);
            }
            document.getElementById('infoFeatCount').innerText = state.selectedFeatures.length;
        });

        container.appendChild(label);
    });
}

async function runClustering() {
    if (state.selectedFeatures.length === 0) {
        alert('Please select at least 1 feature for clustering.');
        return;
    }

    document.getElementById('statusText').innerText = 'Fitting K-Means...';

    try {
        const res = await fetch('/api/cluster', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                n_clusters: state.k,
                features: state.selectedFeatures,
                scaling: state.scaling
            })
        });

        const data = await res.json();

        if (data.status === 'success') {
            state.clusterData = data;
            state.tableData = data.sample_table;

            // Update Header & Metrics
            document.getElementById('statusText').innerText = 'Model Fit & Segmented';
            document.getElementById('metricK').innerText = data.n_clusters;
            document.getElementById('metricSilhouette').innerText = data.silhouette_score.toFixed(3);
            document.getElementById('metricInertia').innerText = Math.round(data.inertia).toLocaleString();
            document.getElementById('metricTotalCustomers').innerText = state.columnInfo.total_rows;

            // Render Visualizations & Content
            renderPCAPlot();
            renderFeatureMeansChart();
            renderPersonaCards(data.personas);
            renderDataTable(data.sample_table);

            // Also trigger evaluation charts in tab 2 silently
            runOptimalKEvaluation(false);
        }
    } catch (err) {
        console.error('Error running clustering:', err);
        document.getElementById('statusText').innerText = 'Error fitting model';
    }
}

// --------------------------------------------------------------------------
// Visualizations (Plotly & Chart.js)
// --------------------------------------------------------------------------

function renderPCAPlot() {
    if (!state.clusterData || !state.clusterData.pca) return;

    const pca = state.clusterData.pca;
    const profiles = state.clusterData.profiles.profiles;

    const traces = [];
    const n_clusters = state.clusterData.n_clusters;

    // Build scatter traces per cluster
    for (let c = 0; c < n_clusters; c++) {
        const indices = [];
        for (let i = 0; i < pca.pca_x.length; i++) {
            // Map customer to cluster label from table if available
            indices.push(i);
        }

        const color = CLUSTER_COLORS[c % CLUSTER_COLORS.length];

        if (state.currentVizMode === '2D') {
            traces.push({
                x: pca.pca_x,
                y: pca.pca_y,
                mode: 'markers',
                type: 'scatter',
                name: `Cluster ${c}`,
                marker: {
                    size: 8,
                    color: color,
                    opacity: 0.85
                }
            });
        } else {
            traces.push({
                x: pca.pca_x,
                y: pca.pca_y,
                z: pca.pca_z || pca.pca_x.map(() => 0),
                mode: 'markers',
                type: 'scatter3d',
                name: `Cluster ${c}`,
                marker: {
                    size: 5,
                    color: color,
                    opacity: 0.8
                }
            });
        }
    }

    // Add Centroids
    if (state.currentVizMode === '2D') {
        traces.push({
            x: pca.centroid_pca_x,
            y: pca.centroid_pca_y,
            mode: 'markers',
            type: 'scatter',
            name: 'Centroids ⭐',
            marker: {
                size: 16,
                color: '#ffffff',
                symbol: 'star',
                line: { color: '#000', width: 2 }
            }
        });
    } else {
        traces.push({
            x: pca.centroid_pca_x,
            y: pca.centroid_pca_y,
            z: pca.centroid_pca_z || pca.centroid_pca_x.map(() => 0),
            mode: 'markers',
            type: 'scatter3d',
            name: 'Centroids ⭐',
            marker: {
                size: 10,
                color: '#ffffff',
                symbol: 'diamond',
                line: { color: '#000', width: 2 }
            }
        });
    }

    const layout = {
        paper_bgcolor: 'transparent',
        plot_bgcolor: 'transparent',
        font: { color: '#94a3b8', family: 'Inter' },
        margin: { l: 40, r: 40, t: 30, b: 40 },
        legend: { orientation: 'h', y: -0.15 },
        xaxis: { title: 'PCA Component 1', gridcolor: 'rgba(255,255,255,0.06)' },
        yaxis: { title: 'PCA Component 2', gridcolor: 'rgba(255,255,255,0.06)' }
    };

    if (state.currentVizMode === '3D') {
        layout.scene = {
            xaxis: { title: 'PCA 1', gridcolor: 'rgba(255,255,255,0.1)' },
            yaxis: { title: 'PCA 2', gridcolor: 'rgba(255,255,255,0.1)' },
            zaxis: { title: 'PCA 3', gridcolor: 'rgba(255,255,255,0.1)' },
            bgcolor: 'transparent'
        };
    }

    Plotly.newPlot('plotlyClusterContainer', traces, layout, { responsive: true, displayModeBar: false });

    // Update variance note
    const varRatio = pca.explained_variance || [];
    const totalVar = ((varRatio.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
    document.getElementById('pcaVarText').innerHTML = `<i class="fa-solid fa-info-circle"></i> PCA components explain ${totalVar}% of overall variance across selected features.`;
}

function renderFeatureMeansChart() {
    if (!state.clusterData) return;

    const ctx = document.getElementById('chartFeatureMeans').getContext('2d');
    const profiles = state.clusterData.profiles.profiles;
    const overall = state.clusterData.profiles.overall_means;
    const features = state.clusterData.feature_cols;

    if (chartMeansInstance) chartMeansInstance.destroy();

    const datasets = profiles.map((p, idx) => ({
        label: `Cluster ${p.cluster_id} (${p.percentage}%)`,
        data: features.map(f => p.means[f] || 0),
        backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length],
        borderRadius: 4
    }));

    // Add overall average bar
    datasets.push({
        label: 'Overall Population Mean',
        data: features.map(f => overall[f] || 0),
        backgroundColor: 'rgba(255, 255, 255, 0.25)',
        borderColor: '#fff',
        borderWidth: 1,
        borderRadius: 4
    });

    chartMeansInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: features,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#94a3b8' } },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

async function runOptimalKEvaluation(switchTabFlag = false) {
    try {
        const res = await fetch('/api/elbow_silhouette', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                features: state.selectedFeatures,
                scaling: state.scaling
            })
        });

        const data = await res.json();

        // 1. Elbow Chart
        const ctxElbow = document.getElementById('chartElbow').getContext('2d');
        if (chartElbowInstance) chartElbowInstance.destroy();

        chartElbowInstance = new Chart(ctxElbow, {
            type: 'line',
            data: {
                labels: data.k_values,
                datasets: [{
                    label: 'WCSS (Inertia)',
                    data: data.wcss,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 6,
                    pointHoverRadius: 9,
                    pointBackgroundColor: '#8b5cf6'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Number of Clusters (K)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'WCSS', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                }
            }
        });

        // 2. Silhouette Chart
        const ctxSil = document.getElementById('chartSilhouette').getContext('2d');
        if (chartSilhouetteInstance) chartSilhouetteInstance.destroy();

        chartSilhouetteInstance = new Chart(ctxSil, {
            type: 'line',
            data: {
                labels: data.k_values,
                datasets: [{
                    label: 'Silhouette Score',
                    data: data.silhouette_scores,
                    borderColor: '#06b6d4',
                    backgroundColor: 'rgba(6, 182, 212, 0.15)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 6,
                    pointHoverRadius: 9,
                    pointBackgroundColor: '#06b6d4'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { title: { display: true, text: 'Number of Clusters (K)', color: '#94a3b8' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'Silhouette Score', color: '#94a3b8' }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    } catch (err) {
        console.error('Error calculating optimal K:', err);
    }
}

// --------------------------------------------------------------------------
// Persona Cards & Classifier
// --------------------------------------------------------------------------

function renderPersonaCards(personas) {
    const container = document.getElementById('personasGrid');
    container.innerHTML = '';

    Object.values(personas).forEach((p, idx) => {
        const card = document.createElement('div');
        
        let cardClass = 'persona-card glass-card';
        if (p.persona_title.includes('VIP')) cardClass += ' vip';
        else if (p.persona_title.includes('Savers')) cardClass += ' savers';
        else if (p.persona_title.includes('Impulse') || p.persona_title.includes('Trend')) cardClass += ' impulse';
        else if (p.persona_title.includes('Budget')) cardClass += ' budget';
        else cardClass += ' mainstream';

        card.className = cardClass;

        card.innerHTML = `
            <div>
                <div class="persona-header">
                    <div>
                        <div class="persona-title">${p.persona_title}</div>
                        <div class="persona-badge">${p.tier}</div>
                    </div>
                    <span class="badge" style="background: ${CLUSTER_COLORS[p.cluster_id % CLUSTER_COLORS.length]}">
                        Cluster ${p.cluster_id} (${p.customer_percentage}%)
                    </span>
                </div>

                <div class="persona-demo"><i class="fa-solid fa-chart-pie"></i> ${p.demographic_summary}</div>
                <p class="persona-desc">${p.description}</p>

                <div class="persona-section-title"><i class="fa-solid fa-lightbulb"></i> Recommended Marketing Strategy</div>
                <div class="persona-strategy">${p.recommended_strategy}</div>

                <div class="persona-section-title"><i class="fa-solid fa-paper-plane"></i> Preferred Channels</div>
                <div class="persona-channels">${p.preferred_channels}</div>
            </div>

            <div class="info-row mt-2" style="border-top: 1px solid var(--border-glass); padding-top: 0.5rem;">
                <span>Risk Status:</span>
                <strong style="color: var(--pink-accent)">${p.risk_level}</strong>
            </div>
        `;

        container.appendChild(card);
    });
}

function renderPredictorFields() {
    const container = document.getElementById('predictorDynamicFields');
    container.innerHTML = '';

    state.selectedFeatures.forEach(feat => {
        const group = document.createElement('div');
        group.className = 'control-group';

        const sampleMean = state.clusterData ? state.clusterData.profiles.overall_means[feat] || 40 : 40;

        group.innerHTML = `
            <label class="input-label">${feat}</label>
            <input type="number" step="any" name="${feat}" value="${sampleMean}" class="form-control" required>
        `;
        container.appendChild(group);
    });
}

async function handlePrediction(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const featureValues = {};

    state.selectedFeatures.forEach(f => {
        featureValues[f] = parseFloat(formData.get(f));
    });

    try {
        const res = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ features: featureValues })
        });
        const data = await res.json();

        if (data.predicted_cluster !== undefined) {
            const p = data.persona;
            const resCard = document.getElementById('predictorResult');
            resCard.innerHTML = `
                <div style="width: 100%">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <span class="badge" style="background: ${CLUSTER_COLORS[data.predicted_cluster % CLUSTER_COLORS.length]}; font-size: 1rem; padding: 0.4rem 1rem;">
                            Predicted Segment: Cluster ${data.predicted_cluster}
                        </span>
                        <span class="text-sub">Centroid Distance: ${data.centroid_distance.toFixed(3)}</span>
                    </div>

                    <h3 style="font-family: var(--font-heading); font-size: 1.4rem; color: var(--cyan-accent); margin-bottom: 0.5rem;">
                        ${p.persona_title}
                    </h3>
                    <p style="color: var(--text-secondary); margin-bottom: 1rem;">${p.description}</p>

                    <div class="persona-section-title"><i class="fa-solid fa-bullseye"></i> Targeted Playbook Action</div>
                    <div class="persona-strategy" style="font-size: 0.95rem;">${p.recommended_strategy}</div>
                </div>
            `;
        }
    } catch (err) {
        console.error('Error making prediction:', err);
    }
}

// --------------------------------------------------------------------------
// Data Table & CSV Export
// --------------------------------------------------------------------------

function renderDataTable(data) {
    if (!data || data.length === 0) return;

    const thead = document.getElementById('tableHeader');
    const tbody = document.getElementById('tableBody');

    const columns = Object.keys(data[0]);

    thead.innerHTML = `
        <tr>
            ${columns.map(col => `<th>${col}</th>`).join('')}
        </tr>
    `;

    tbody.innerHTML = data.map(row => `
        <tr>
            ${columns.map(col => {
                if (col === 'Cluster') {
                    const color = CLUSTER_COLORS[row[col] % CLUSTER_COLORS.length];
                    return `<td><span class="cluster-pill" style="background: ${color}">Cluster ${row[col]}</span></td>`;
                }
                return `<td>${row[col]}</td>`;
            }).join('')}
        </tr>
    `).join('');
}

function filterTable(searchTerm) {
    const term = searchTerm.toLowerCase();
    const filtered = state.tableData.filter(row => {
        return Object.values(row).some(val => String(val).toLowerCase().includes(term));
    });
    renderDataTable(filtered);
}

function exportCSV() {
    if (!state.tableData || state.tableData.length === 0) return;

    const keys = Object.keys(state.tableData[0]);
    let csvContent = keys.join(',') + '\n';

    state.tableData.forEach(row => {
        const values = keys.map(k => `"${row[k]}"`);
        csvContent += values.join(',') + '\n';
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `segmented_customers_k${state.k}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
