// Columns to display and their headers
const COLUMNS = [
  { key: 'asset', header: 'Asset' },
  { key: 'price', header: 'Price' },
  { key: 'ob_metric', header: 'OB Metric' },
  { key: 'divergence_metric', header: 'Divergence Metric' },
  { key: 'iob_top', header: 'OB Top' },
  { key: 'iob_btm', header: 'OB Bottom' },
  { key: 'iob_left_utc', header: 'OB Start (UTC)' },
  { key: 'mfi_bd', header: 'MFI BD' },
  { key: 'stoch_bd', header: 'Stoch BD' },
  { key: 'rsi_bd', header: 'RSI BD' },
  { key: 'williams_r_bd', header: 'W%R BD' },
];

function renderBoolArray(arr) {
  if (!Array.isArray(arr)) return '<span>-</span>';
  return arr
    .map(b => `<span class="bd-${b}">${b ? 'T' : 'F'}</span>`)
    .join(' ');
}

function getMetricColorClass(metric) {
  if (metric >= 0.7) return 'high';
  if (metric >= 0.4) return 'medium';
  return 'low';
}

function renderMetric(metric) {
  const rounded = (metric * 100).toFixed(1);
  const colorClass = getMetricColorClass(metric);
  return `
    <div class="d-flex align-items-center">
      <span class="me-2" style="width: 40px;">${rounded}%</span>
      <div class="metric-bar">
        <div class="metric-bar-inner ${colorClass}" style="width: ${rounded}%;"></div>
      </div>
    </div>
  `;
}

function renderTable(containerId, data) {
  const container = document.getElementById(containerId);

  if (data.length === 0) {
    container.innerHTML = '<p class="text-muted">No signals found.</p>';
    return;
  }

  let table = '<table class="table table-dark table-striped table-hover">';

  // Header
  table += '<thead><tr>';
  for (const col of COLUMNS) table += `<th>${col.header}</th>`;
  table += '</tr></thead>';

  // Body
  table += '<tbody>';
  for (const row of data) {
    table += '<tr>';
    for (const col of COLUMNS) {
      let cell = row[col.key];

      if (col.key === 'ob_metric' || col.key === 'divergence_metric') {
        cell = renderMetric(cell);
      } else if (col.key.endsWith('_bd')) {
        cell = renderBoolArray(cell);
      } else if (col.key === 'iob_left_utc') {
        cell = new Date(cell).toLocaleString('sv-SE', { timeZone: 'UTC' }).replace('T', ' ');
      } else if (typeof cell === 'number' && !['price', 'iob_top', 'iob_btm'].includes(col.key)) {
        cell = cell.toFixed(2);
      }

      table += `<td>${cell}</td>`;
    }
    table += '</tr>';
  }
  table += '</tbody></table>';

  container.innerHTML = table;
}

async function fetchData() {
  const loading = document.getElementById('loading');
  loading.style.display = 'block';

  try {
    const response = await fetch('/api/signals');
    const data = await response.json();

    renderTable('long-signals-container', data.long);
    renderTable('short-signals-container', data.short);

    const lastUpdatedEl = document.getElementById('last-updated');
    if (data.last_updated) {
      const local = new Date(data.last_updated + 'Z').toLocaleString();
      lastUpdatedEl.textContent = `Last Updated: ${local}`;
    } else {
      lastUpdatedEl.textContent = 'Last Updated: Never';
    }

  } catch (err) {
    console.error(err);
  } finally {
    loading.style.display = 'none';
  }
}

document.addEventListener('DOMContentLoaded', fetchData);
// setInterval(fetchData, 60000);
