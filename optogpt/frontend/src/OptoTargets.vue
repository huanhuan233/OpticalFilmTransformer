<!-- /src/OptoTargets.vue (合并修BUG版，可直接复制替换) --> 
<template>
  <div class="page">
    <!-- 顶部标题栏 -->
    <va-card class="mb-3">
      <va-card-content>
        <div class="title-row">
          <div>
            <div class="title">光学目标与权重输入</div>
            <div class="subtitle">逐行添加：通道（R/T） + 波长（380–750，步长 5） + 数值（0–1 或 0–100%） + 权重</div>
          </div>
          <div class="title-actions">
            <va-button size="small" preset="primary" icon="auto_awesome" @click="loadDemo('basic')">填入示例A</va-button>
            <va-button size="small" preset="secondary" icon="tune" @click="loadDemo('notch')">填入示例B</va-button>
            <va-button size="small" preset="secondary" icon="science" @click="loadDemo('absorber')">填入示例C</va-button>
          </div>
        </div>
      </va-card-content>
    </va-card>

    <va-row :gutter="16" class="mb-3">
      <!-- 左列：表单编辑区 + 结果 -->
      <va-col :xs="12" :md="7">
        <va-card>
          <va-card-title>指令行编辑器</va-card-title>
          <va-card-content>
            <!-- 行编辑表格 -->
            <div class="table-scroll max5">
              <table class="grid-table">
                <thead>
                  <tr>
                    <th style="width:88px">通道</th>
                    <th style="width:120px">波长 (nm)</th>
                    <th style="width:120px">目标值</th>
                    <th style="width:100px">权重</th>
                    <th style="width:90px; text-align:center">操作</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, idx) in rows" :key="row.id">
                    <td>
                      <va-select dense :options="channelOptions" v-model="row.ch" placeholder="R/T" value-by="value" text-by="text" />
                    </td>
                    <td>
                      <va-select dense :options="lambdaOptions" v-model="row.lam" placeholder="波长" searchable />
                    </td>
                    <td>
                      <va-input dense type="number" v-model.number="row.val" placeholder="0.8 或 80" />
                    </td>
                    <td>
                      <va-input dense type="number" v-model.number="row.w" placeholder="1" />
                    </td>
                    <td class="center">
                      <va-button size="small" color="danger" outline icon="delete" @click="removeRow(idx)">删除</va-button>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div class="line-actions">
              <!-- 新增：导入按钮 + 隐藏文件选择框（CSV/Excel） -->
              <va-button icon="upload_file" @click="pickFile">导入 CSV/Excel</va-button>
              <input ref="fileInput" type="file" accept=".csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,application/vnd.ms-excel" class="hidden" @change="handleImport" />

              <va-button icon="add" @click="addRow()">新增一行</va-button>
              <va-button color="danger" outline icon="delete_sweep" @click="clearRows()">清空</va-button>
              <va-chip v-if="rows.length" color="info" outline>共 {{ rows.length }} 行</va-chip>
            </div>

            <va-divider class="my-3" />

            <va-row :gutter="12" class="mb-1">
              <va-col :xs="12" :md="6">
                <div class="section-title">Top-kp 数量</div>
                <va-select v-model="topKpCount" :options="topKpOptions" text-by="text" value-by="value" dense />
                <div class="section-title mt-3">权重策略</div>
                <div class="weight-control-row">
                  <va-select v-model="weightMode" :options="weightModeOptions" text-by="text" value-by="value" dense />
                  <va-switch v-model="useGaussian" label="使用高斯扩散权重" dense />
                </div>
                <div v-if="useGaussian" class="two-col">
                  <va-input dense type="number" v-model.number="sigmaNm" label="Gaussian σ (nm)" />
                  <va-input dense type="number" v-model.number="gaussBase" label="基线 base" />
                </div>
              </va-col>

              <va-col :xs="12" :md="6">
                <div class="section-title">TFCalc 参数</div>
                <div class="two-col">
                  <va-input dense type="number" v-model.number="tfTol" label="Tol 基准（默认 0.05）" />
                  <va-input dense type="number" v-model.number="tfK" label="k 指数（1/2/4/8…）" />
                </div>
                <!-- <va-select dense :options="[{ text: 'N=1 / I=1 / D=1（默认）', value: 'const' }]" v-model="tfPreset" class="mt-1" /> -->
              </va-col>
            </va-row>

            <!-- <va-alert v-if="parseError" color="danger" outline icon="error_outline">{{ parseError }}</va-alert>
            <va-alert v-else color="success" outline icon="task_alt">解析成功</va-alert> -->

            <div class="submit-row">
              <!-- <va-button preset="primary" icon="check" @click="emitPayload">生成并发送</va-button>
              <va-button preset="secondary" icon="content_copy" @click="copyJSON">复制 JSON</va-button>
              <va-button preset="secondary" icon="download" @click="downloadJSON">下载 JSON</va-button> -->
              <va-spacer />
              <va-button color="primary" icon="rocket_launch" @click="runBackend">提交到后端计算</va-button>
            </div>

            <!-- 简要回显（兼容老返回） -->
            <va-alert v-if="backendResp && (!backendResp.samples || !backendResp.samples.length)" color="info" outline icon="analytics" class="mt-2">
              <div><b>后端结果：</b></div>
              <div class="result-line">最优结构：<code class="code">{{ (backendResp.structure ?? []).join(', ') }}</code></div>
              <div class="result-line" v-if="backendResp.tf_score !== undefined">TFCalc 分数：<b>{{ n6(backendResp.tf_score) }}</b></div>
              <div class="result-line" v-if="backendResp.mae_weight !== undefined">加权 MAE：{{ n6(backendResp.mae_weight) }}</div>
              <div class="result-line" v-if="backendResp.error">错误：{{ backendResp.error }}</div>
            </va-alert>

            <!-- Top-kp 列表展示 -->
            <va-divider class="my-3" />
            <va-card outlined>
              <va-card-title>
                <div style="display: flex; justify-content: space-between; align-items: center; width: 100%;">
                  <span>Top-kp 采样结果</span>
                  <va-switch v-model="enableOptimization" label="优化结果" dense />
                </div>
              </va-card-title>
              <va-card-content>
                <div v-if="sampleTable.length === 0" class="text-muted">暂无结果，请点击“提交到后端计算”。</div>

                <div v-else class="table-scroll">
                  <table class="grid-table">
                    <thead>
                      <tr>
                        <th style="width:80px">编号</th>
                        <th style="width:120px">TF 分数</th>
                        <th style="width:100px">层数</th>
                        <th>膜系（tokens）</th>
                        <th style="width:90px">查看</th>
                      </tr>
                    </thead>
                        <tbody>
                        <tr v-for="(s, i) in sampleTable" :key="i">
                            <!-- 用 i 来显示编号 -->
                            <td>#{{ String(i).padStart(2, '0') }}</td>
                            <td><b>{{ n6(s.tf) }}</b></td>
                            <td>{{ parseTokens(s.structure).length }}</td>
                            <td class="mono">{{ s.structure.join(', ') }}</td>
                            <td class="center">
                            <va-checkbox 
                              :model-value="selectedSamples.includes(i)" 
                              @update:model-value="(val: boolean) => onSampleSelectionChange(i, val)" 
                            />
                            </td>
                        </tr>
                        </tbody>
                  </table>
                </div>

                <div v-if="bestOne" class="best-box">
                  <div class="best-title">最佳（TFCalc）</div>
                  <div>来源：<b>{{ bestOne?.source ?? 'N/A' }}</b>；分数：<b>{{ bestOne?.score !== undefined ? n6(bestOne!.score!) : 'N/A' }}</b></div>
                  <div class="mono mt-1">{{ (bestOne?.structure ?? []).join(', ') }}</div>
                  <!-- R/T 光谱图 -->
                  <div v-if="bestSpectrum && bestOne" class="mt-3">
                    <div class="block-title">R/T 光谱（TMM计算）</div>
                    <div v-if="enableOptimization && !optimizedSpectrum" class="mb-2" style="color: #6b7280; font-size: 12px;">
                      正在优化膜系结构...
                    </div>
                    <div style="display: flex; gap: 16px;">
                      <div style="flex: 1; overflow-x: auto;">
                        <canvas 
                          ref="spectrumCanvas" 
                          class="spectrum-canvas" 
                          @mousemove="onCanvasMouseMove" 
                          @mouseleave="onCanvasMouseLeave"
                        ></canvas>
                        <div v-if="hoveredCurve" class="mt-2" style="font-size: 12px; color: #059669; text-align: center;">
                          {{ hoveredCurve }}
                        </div>
                      </div>
                      <div ref="legendContainer" class="spectrum-legend"></div>
                    </div>
                    <!-- 显示所有优化后的结构 -->
                    <div v-if="enableOptimization" class="mt-2" style="font-size: 12px;">
                      <!-- 最佳结果的优化后结构 -->
                      <div v-if="optimizedSpectrum" style="color: #059669; margin-bottom: 8px;">
                        <div><b>最佳-优化后结构：</b>{{ optimizedSpectrum.optimized_structure.join(', ') }}</div>
                      </div>
                      <!-- 选中采样结果的优化后结构 -->
                      <template v-for="sampleIdx in selectedSamples" :key="sampleIdx">
                        <div v-if="getOptimizedStructure(sampleIdx)" style="color: #059669; margin-bottom: 8px;">
                          <div><b>#{{ String(sampleIdx).padStart(2, '0') }}-优化后结构：</b>{{ getOptimizedStructure(sampleIdx) }}</div>
                        </div>
                      </template>
                    </div>
                  </div>
                  <div v-else-if="bestOne" class="mt-2" style="color: #6b7280; font-size: 12px;">
                    正在计算R/T光谱...
                  </div>
                </div>
              </va-card-content>
            </va-card>

            <!-- 弹窗：层材料/厚度表 -->
            <!-- <va-modal v-model="layerDlg.open" title="膜系层表" size="large">
              <template #message>
                <div v-if="layerDlg.rows.length === 0" class="text-muted">无可显示数据</div>
                <div v-else class="table-scroll">
                  <table class="grid-table">
                    <thead>
                      <tr>
                        <th style="width:80px">序号</th>
                        <th style="width:140px">材料</th>
                        <th style="width:140px">厚度 (nm)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr v-for="(r,i) in layerDlg.rows" :key="i">
                        <td>{{ i+1 }}</td>
                        <td>{{ r.material }}</td>
                        <td>{{ r.thickness }}</td>
                      </tr>
                    </tbody>
                  </table>
                  <div class="mono mt-2">原始 tokens：{{ layerDlg.tokens.join(', ') }}</div>
                  <div class="mt-1">TF 分数：<b>{{ layerDlg.tf !== undefined ? n6(layerDlg.tf) : 'N/A' }}</b></div>
                </div>
              </template>
              <template #footer>
                <va-button preset="primary" @click="exportCSV">导出 CSV</va-button>
                <va-spacer />
                <va-button preset="secondary" @click="layerDlg.open=false">关闭</va-button>
              </template>
            </va-modal> -->

          </va-card-content>
        </va-card>
      </va-col>

      <!-- 右列：预览 / 数据 / 曲线 -->
      <!-- <va-col :xs="12" :md="5">
        <va-card>
          <va-card-title>预览 / 数据 / 曲线</va-card-title>
          <va-card-content>
            <va-tabs v-model="tab" grow>
              <template #tabs>
                <va-tab name="preview">预览</va-tab>
                <va-tab name="table">数据</va-tab>
                <va-tab name="plot">曲线</va-tab>
              </template>
            </va-tabs>

            <div v-if="tab==='preview'" class="preview">
              <div class="preview-block">
                <div class="block-title">指令文本（自动生成）</div>
                <div class="mono-box">{{ directives }}</div>
              </div>
              <div class="two-col">
                <div class="preview-block">
                  <div class="block-title">R 目标（前 10 项）</div>
                  <div class="mono-box small">{{ R_target.slice(0,10).map(n4).join(', ') }}</div>
                </div>
                <div class="preview-block">
                  <div class="block-title">T 目标（前 10 项）</div>
                  <div class="mono-box small">{{ T_target.slice(0,10).map(n4).join(', ') }}</div>
                </div>
              </div>
              <div class="preview-block">
                <div class="block-title">权重（拼接 [wR..., wT...]，前 20 项）</div>
                <div class="mono-box small">{{ spec_weights.slice(0,20).map(n4).join(', ') }}</div>
              </div>
            </div>

            <div v-else-if="tab==='table'" class="table-scroll">
              <table class="grid-table">
                <thead>
                  <tr>
                    <th>λ (nm)</th>
                    <th>R</th>
                    <th>T</th>
                    <th>wR</th>
                    <th>wT</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(lam, i) in lamNm" :key="lam">
                    <td>{{ lam }}</td>
                    <td>{{ n4(R_target[i]) }}</td>
                    <td>{{ n4(T_target[i]) }}</td>
                    <td>{{ n4(spec_weights[i]) }}</td>
                    <td>{{ n4(spec_weights[i + lamNm.length]) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div v-else class="plot-grid">
              <div>
                <div class="block-title">R/T 目标</div>
                <canvas ref="rtCanvas" class="plot"></canvas>
              </div>
              <div>
                <div class="block-title">权重（wR / wT）</div>
                <canvas ref="wCanvas" class="plot small"></canvas>
              </div>
            </div>
          </va-card-content>
        </va-card>
      </va-col> -->
    </va-row>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUpdated, nextTick } from 'vue'


const __envBase = (import.meta as any).env?.VITE_API_BASE
const __winBase = (window as any).__API_BASE__
const API_BASE = (__envBase !== undefined)
  ? __envBase // 允许设置为空字符串 ''，以便 fetch(`${API_BASE}/api/...`) -> '/api/...'
  : (__winBase !== undefined ? __winBase : `${location.protocol}//${location.hostname}:8174`)

/* ---------- 类型 ---------- */
type Channel = 'R' | 'T'
interface Row { id: string; ch: Channel; lam: number; val: number | null; w: number | null }
interface SampleItem { idx: number; tf: number; structure: string[]; tag?: string }
interface BestItem { source?: string; score?: number; structure?: string[] }
interface BackendResp {
  structure?: string[]; tf_score?: number; mae_weight?: number; error?: string;
  samples?: SampleItem[]; best?: BestItem;
}

/* ---------- λ 网格 / 选项 ---------- */
function makeWavelengths (): number[] { const out: number[] = []; for (let x = 380; x <= 750 + 1e-9; x += 5) out.push(Math.round(x)); return out }
const lamNm = ref<number[]>(makeWavelengths())
const lambdaOptions = computed(() => lamNm.value.map(v => ({ text: String(v), value: v })))
const channelOptions = [{ text: 'R', value: 'R' }, { text: 'T', value: 'T' }]

/* ---------- 行编辑状态 ---------- */
const uid = () => Math.random().toString(36).slice(2, 9)
const rows = ref<Row[]>([
  { id: uid(), ch: 'R', lam: 385, val: 0.42, w: 2 },
  { id: uid(), ch: 'T', lam: 550, val: 0.80, w: 4 },
  { id: uid(), ch: 'R', lam: 700, val: 0.30, w: 3 },
])
function addRow () { rows.value.push({ id: uid(), ch: 'R', lam: 550, val: 0.8, w: 1 }) }
function removeRow (i: number) { rows.value.splice(i, 1) }
function clearRows () { rows.value = [] }

/* ---------- 导入 CSV / Excel ---------- */
const fileInput = ref<HTMLInputElement | null>(null)
function pickFile () { fileInput.value?.click() }

async function handleImport (ev: Event) {
  const el = ev.target as HTMLInputElement
  const f = el.files?.[0]
  if (!f) return
  try {
    const ext = (f.name.split('.').pop() || '').toLowerCase()
    let recs: any[] = []
    if (ext === 'csv') recs = await parseCSVFile(f)
    else recs = await parseXLSXFile(f)
    const mapped = normalizeImportedRows(recs)
    if (!mapped.length) throw new Error('未在文件中解析到有效数据')
    rows.value = mapped
    recompute(); safeDraw()
  } catch (e: any) {
    parseError.value = `导入失败：${String(e?.message ?? e)}`
    console.error('[import] 解析失败', e)
  } finally {
    if (fileInput.value) fileInput.value.value = ''
  }
}
async function parseCSVFile (file: File): Promise<any[]> {
  const text = await file.text()
  const lines = text.split(/\r?\n/).filter(s => s.trim() !== '')
  if (!lines.length) return []

  // 自动识别分隔符：优先制表符，其次逗号/分号
  const split = (s: string) => {
    if (/\t/.test(s)) return s.split('\t')
    return s.split(/,|;/)
  }

  const first = split(lines[0]).map(s => s.trim())
  const looksLikeHeader =
    first.some(v => /^(R|T)$/i.test(v)) ? false :                 // 第一格像数据：R/T => 无表头
    first.some(v => /^λ|lam|wavelength|波长|channel|通道|value|目标/.test(v)) || // 有典型字段名
    first.some(v => /[a-zA-Z]/.test(v))                           // 或至少含字母
  // 有表头：生成对象；无表头：按 [ch,lam,val,w]
  const rows = lines.slice(looksLikeHeader ? 1 : 0).map(line => {
    const cells = split(line).map(s => s.trim())
    if (looksLikeHeader) {
      const obj: any = {}
      first.forEach((h, i) => { obj[h] = cells[i] ?? '' })
      return obj
    } else {
      return { ch: cells[0], lam: cells[1], val: cells[2], w: cells[3] }
    }
  })
  return rows
}

async function parseXLSXFile (file: File): Promise<any[]> {
  const buf = await file.arrayBuffer()
  let XLSX: any
  try { XLSX = await import('xlsx') } catch { throw new Error('需要依赖 xlsx，请先执行 npm i xlsx') }
  const wb = XLSX.read(buf, { type: 'array' })
  const first = wb.SheetNames?.[0]; if (!first) return []
  const ws = wb.Sheets[first]

  // 读取为二维数组，方便判断是否有表头
  const arr: any[][] = XLSX.utils.sheet_to_json(ws, { header: 1, defval: '' })
  if (!arr.length) return []

  const head = (arr[0] || []).map(String)
  const looksLikeHeader =
    head.some(v => /^(R|T)$/i.test(v)) ? false :
    head.some(v => /^λ|lam|wavelength|波长|channel|通道|value|目标/.test(v)) ||
    head.some(v => /[a-zA-Z]/.test(v))

  if (looksLikeHeader) {
    // 直接让 xlsx 生成对象
    return XLSX.utils.sheet_to_json(ws, { defval: '' }) as any[]
  } else {
    // 无表头：按前四列 [ch,lam,val,w]
    return arr.map(r => ({ ch: r[0], lam: r[1], val: r[2], w: r[3] }))
  }
}
function normalizeImportedRows (recs: any[]): Row[] {
  const alias = {
    ch:  ['ch','channel','通道','r/t','rt'],
    lam: ['lam','wavelength','λ','波长','wl'],
    val: ['val','value','target','目标','目标值'],
    w:   ['w','weight','权重']
  }
  const pickKey = (obj: any, names: string[]) => {
    const keys = Object.keys(obj)
    const lower = keys.map(k => k.toLowerCase())
    for (const n of names) {
      const i = lower.indexOf(String(n).toLowerCase())
      if (i >= 0) return keys[i]
    }
    return null
  }

  const out: Row[] = []
  for (const r of recs) {
    // 同时兼容 “对象记录” 与 “无表头按位记录”
    const isArrayLike = Array.isArray(r) || (r && typeof r === 'object' && '0' in r)
    const rec: any = isArrayLike ? { ch: r[0], lam: r[1], val: r[2], w: r[3] } : r

    const kCh  = pickKey(rec, alias.ch)  ?? 'ch'
    const kLam = pickKey(rec, alias.lam) ?? 'lam'
    const kVal = pickKey(rec, alias.val) ?? 'val'
    const kW   = pickKey(rec, alias.w)   ?? 'w'

    let ch = String(rec[kCh]).trim().toUpperCase()
    if (ch !== 'R' && ch !== 'T') continue

    let lam = Number(rec[kLam]); if (!Number.isFinite(lam)) continue
    lam = Math.round(lam); lam = Math.round(lam / 5) * 5

    let val = Number(rec[kVal]); if (!Number.isFinite(val)) continue
    if (val > 1) val = val / 100; val = Math.max(0, Math.min(1, val))

    let w = Number(rec[kW]); if (!Number.isFinite(w)) w = 0; w = Math.max(0, w)

    out.push({ id: uid(), ch: ch as Channel, lam, val, w })
  }
  return out
}

/* ---------- Top-kp 数量 ---------- */
const topKpCount = ref<number>(20)
const topKpOptions = computed(() => {
  const options: Array<{ text: string; value: number }> = []
  for (let i = 5; i <= 50; i++) {
    options.push({ text: String(i), value: i })
  }
  return options
})

/* ---------- 权重 / TFCalc ---------- */
const weightMode = ref<'fullband' | 'anchors'>('fullband')
const weightModeOptions = [
  { text: '全谱默认 = 1（未指定=1）', value: 'fullband' },
  { text: '仅锚点计分（未指定=0）', value: 'anchors' },
]
const useGaussian = ref<boolean>(false)
const sigmaNm = ref<number>(15)
const gaussBase = ref<number>(0)
const tfTol = ref<number>(0.05)
const tfK = ref<number>(2)
const tfPreset = ref<'const'>('const')

/* ---------- 指令文本 ---------- */
const directives = ref<string>('')
const parseError = ref<string>('')

function rowsToDirectives (): string {
  const sorted = [...rows.value].sort((a, b) => a.lam - b.lam)
  const lines: string[] = []
  for (const r of sorted) {
    if (r.val == null || r.w == null) continue
    let v = Number(r.val)
    if (!Number.isFinite(v)) continue
    // 支持输入 0–1 或 0–100%，>1 的当作百分比
    if (v > 1) v = v / 100
    v = Math.max(0, Math.min(1, v))

    const w = Math.max(0, Number(r.w) || 0)
    // 处理 ch 字段：可能是字符串，也可能是对象（防御性处理）
    let ch: string = typeof r.ch === 'string' ? r.ch : ((r.ch as any)?.value || (r.ch as any)?.text || 'R')
    // 处理 lam 字段：可能是数字，也可能是对象（防御性处理）
    let lam: number = typeof r.lam === 'number' ? r.lam : ((r.lam as any)?.value || (r.lam as any)?.text || 550)
    lines.push(`${ch},${lam},${v},${w}`)
  }
  // 关键：逐行换行（并加一个结尾换行，防止最后一行也被黏连）
  return lines.join('\n') + '\n'
}
watch(rows, () => { directives.value = rowsToDirectives() }, { deep: true, immediate: true })

/* ---------- 目标与权重 ---------- */
const R_target = ref<number[]>([])
const T_target = ref<number[]>([])
const spec_weights = ref<number[]>([])
const N_vec = ref<number[]>([])
const Tol_vec = ref<number[]>([])
const I_vec = ref<number[]>([])
const D_vec = ref<number[]>([])

const clamp01 = (x: number) => Math.max(0, Math.min(1, Number(x)))
function interp1d (xs: number[], ys: number[], xq: number): number {
  if (xq <= xs[0]) return ys[0]
  if (xq >= xs[xs.length - 1]) return ys[ys.length - 1]
  let i = 0; while (i < xs.length - 1 && !(xs[i] <= xq && xq <= xs[i + 1])) i++
  const t = (xq - xs[i]) / (xs[i + 1] - xs[i]); return ys[i] * (1 - t) + ys[i + 1] * t
}
function gaussianWeights (centers: Array<[number, number]>, base = 0, sigma = 15): number[] {
  const wl = lamNm.value; const out = wl.map(() => base)
  for (const [lam, alpha] of centers) { for (let i = 0; i < wl.length; i++) { const g = Math.exp(-0.5 * Math.pow((wl[i] - lam) / sigma, 2)); out[i] = Math.max(out[i], g * alpha) } }
  return out
}
function nearestWavelengthIdx (lam: number): number {
  const wl = lamNm.value; let bi = 0; let bd = Infinity
  for (let i = 0; i < wl.length; i++) { const d = Math.abs(wl[i] - lam); if (d < bd) { bd = d; bi = i } }
  return bi
}

function recompute (): void {
  parseError.value = ''
  try {
    const rPts: Array<[number, number, number]> = []
    const tPts: Array<[number, number, number]> = []
    for (const r of rows.value) {
      if (r.val == null || r.w == null) continue
      let v = Number(r.val); if (v > 1) v = v / 100; v = clamp01(v)
      const a = Math.max(0, Number(r.w)) || 0
      if (r.ch === 'R') rPts.push([r.lam, v, a]); else tPts.push([r.lam, v, a])
    }
    rPts.sort((a, b) => a[0] - b[0]); tPts.sort((a, b) => a[0] - b[0])

    const buildChannel = (pts: Array<[number, number, number]>, def = 0): number[] => {
      const wl = lamNm.value
      if (pts.length === 0) return wl.map(() => def)
      if (pts.length === 1) return wl.map(() => pts[0][1])
      const xs = pts.map(p => p[0]); const ys = pts.map(p => p[1])
      return wl.map(l => clamp01(interp1d(xs, ys, l)))
    }
    R_target.value = buildChannel(rPts, 0)
    T_target.value = buildChannel(tPts, 0)

    const base = (weightMode.value === 'fullband') ? 1 : 0
    let wR: number[] = lamNm.value.map(() => base)
    let wT: number[] = lamNm.value.map(() => base)
    if (useGaussian.value) {
      wR = gaussianWeights(rPts.map(p => [p[0], p[2]]), gaussBase.value, sigmaNm.value)
      wT = gaussianWeights(tPts.map(p => [p[0], p[2]]), gaussBase.value, sigmaNm.value)
    }
    for (const [lam, _v, a] of rPts) wR[nearestWavelengthIdx(lam)] = a
    for (const [lam, _v, a] of tPts) wT[nearestWavelengthIdx(lam)] = a
    spec_weights.value = [...wR, ...wT]

    const Nw = lamNm.value.map(() => 1)
    const Tw = lamNm.value.map(() => tfTol.value)
    for (const [lam, _v, a] of [...rPts, ...tPts]) Tw[nearestWavelengthIdx(lam)] = tfTol.value / Math.max(a, 1e-6)
    N_vec.value = [...Nw, ...Nw]
    Tol_vec.value = [...Tw, ...Tw]
    I_vec.value = [...Nw, ...Nw]
    D_vec.value = [...Nw, ...Nw]

    safeDraw()
  } catch (e: any) { parseError.value = String(e?.message ?? e) }
}
// 修复：避免在未挂载时触发绘制（去掉 immediate）
watch([rows, weightMode, useGaussian, sigmaNm, gaussBase, tfTol], () => recompute(), { deep: true, immediate: false })

/* ---------- 画布（安全绘制） ---------- */
const rtCanvas = ref<HTMLCanvasElement | null>(null)
const wCanvas = ref<HTMLCanvasElement | null>(null)

function drawLine (ctx: CanvasRenderingContext2D, xs: number[], ys: number[]) {
  if (!xs.length) return; ctx.beginPath(); ctx.moveTo(xs[0], ys[0]); for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], ys[i]); ctx.stroke()
}

/* ---------- 绘制R/T光谱图（支持多结果显示） ---------- */
function drawSpectrumChart(): void {
  if (!spectrumCanvas.value || !bestSpectrum.value) return
  
  const c = spectrumCanvas.value
  const ctx = c.getContext('2d') as CanvasRenderingContext2D
  const W = 600
  const H = 400
  
  c.width = Math.max(1, Math.floor(W * devicePixelRatio))
  c.height = Math.max(1, Math.floor(H * devicePixelRatio))
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.scale(devicePixelRatio, devicePixelRatio)
  
  ctx.clearRect(0, 0, W, H)
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, W, H)
  
  const leftPad = 60
  const rightPad = 40
  const topPad = 40
  const bottomPad = 50
  const plotW = W - leftPad - rightPad
  const plotH = H - topPad - bottomPad
  
  const wavelengths = lamNm.value
  const wlMin = wavelengths[0]
  const wlMax = wavelengths[wavelengths.length - 1]
  
  // 绘制网格
  ctx.strokeStyle = '#e5e7eb'
  ctx.lineWidth = 1
  for (let p = 0; p <= 100; p += 20) {
    const y = topPad + plotH * (1 - p / 100)
    ctx.beginPath()
    ctx.moveTo(leftPad, y)
    ctx.lineTo(leftPad + plotW, y)
    ctx.stroke()
  }
  for (let wl = 400; wl <= 750; wl += 50) {
    if (wl >= wlMin && wl <= wlMax) {
      const x = leftPad + plotW * ((wl - wlMin) / (wlMax - wlMin))
      ctx.beginPath()
      ctx.moveTo(x, topPad)
      ctx.lineTo(x, topPad + plotH)
      ctx.stroke()
    }
  }
  
  // 绘制边框
  ctx.strokeStyle = '#000000'
  ctx.lineWidth = 1.5
  ctx.strokeRect(leftPad, topPad, plotW, plotH)
  
  // 坐标轴标签
  ctx.fillStyle = '#000000'
  ctx.font = '12px Arial'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  for (let wl = 400; wl <= 750; wl += 50) {
    if (wl >= wlMin && wl <= wlMax) {
      const x = leftPad + plotW * ((wl - wlMin) / (wlMax - wlMin))
      ctx.fillText(String(wl), x, H - bottomPad + 20)
    }
  }
  ctx.fillText('Wavelength (nm)', W / 2, H - 10)
  
  ctx.textAlign = 'right'
  for (let p = 0; p <= 100; p += 20) {
    const y = topPad + plotH * (1 - p / 100)
    ctx.fillText(String(p), leftPad - 10, y)
  }
  ctx.save()
  ctx.translate(15, H / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.fillText('Percent (%)', 0, 0)
  ctx.restore()
  
  // 绘制最佳结果的初始曲线
  const bestColor = colorPalette[0]
  drawCurve(ctx, bestSpectrum.value.R, bestSpectrum.value.T, wavelengths, wlMin, wlMax, 
            leftPad, topPad, plotW, plotH, bestColor.R, bestColor.T, true, '最佳-初始')
  
  // 绘制最佳结果的优化后曲线
  if (enableOptimization.value && optimizedSpectrum.value) {
    drawCurve(ctx, optimizedSpectrum.value.R, optimizedSpectrum.value.T, wavelengths, wlMin, wlMax,
              leftPad, topPad, plotW, plotH, bestColor.R, bestColor.T, false, '最佳-优化后')
  }
  
  // 绘制选中采样结果的曲线
  const samples = Array.isArray(selectedSamples.value) ? selectedSamples.value : []
  samples.forEach((sampleIdx, colorIdx) => {
    const spectrum = sampleSpectra.value.get(sampleIdx)
    if (!spectrum) return
    
    const color = colorPalette[(colorIdx + 1) % colorPalette.length]
    const label = `#${String(sampleIdx).padStart(2, '0')}`
    
    // 绘制初始曲线
    if (spectrum.initial) {
      drawCurve(ctx, spectrum.initial.R, spectrum.initial.T, wavelengths, wlMin, wlMax,
                leftPad, topPad, plotW, plotH, color.R, color.T, true, `${label}-初始`)
    }
    
    // 绘制优化后曲线
    if (spectrum.optimized) {
      drawCurve(ctx, spectrum.optimized.R, spectrum.optimized.T, wavelengths, wlMin, wlMax,
                leftPad, topPad, plotW, plotH, color.R, color.T, false, `${label}-优化后`)
    }
  })
  
  // 更新外部图例
  updateLegend()
}

function drawCurve(ctx: CanvasRenderingContext2D, R: number[], T: number[], wavelengths: number[],
                  wlMin: number, wlMax: number, leftPad: number, topPad: number, 
                  plotW: number, plotH: number, colorR: string, colorT: string, 
                  isDashed: boolean, label: string) {
  ctx.setLineDash(isDashed ? [5, 5] : [])
  
  // 绘制R曲线
  ctx.strokeStyle = colorR
  ctx.lineWidth = isDashed ? 2 : 2.5
  ctx.beginPath()
  for (let i = 0; i < wavelengths.length; i++) {
    const x = leftPad + plotW * ((wavelengths[i] - wlMin) / (wlMax - wlMin))
    const y = topPad + plotH * (1 - (R[i] * 100) / 100)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
  
  // 绘制T曲线
  ctx.strokeStyle = colorT
  ctx.lineWidth = isDashed ? 2 : 2.5
  ctx.beginPath()
  for (let i = 0; i < wavelengths.length; i++) {
    const x = leftPad + plotW * ((wavelengths[i] - wlMin) / (wlMax - wlMin))
    const y = topPad + plotH * (1 - (T[i] * 100) / 100)
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.stroke()
  
  ctx.setLineDash([])
}

function updateLegend() {
  if (!legendContainer.value) return
  
  const items: Array<{ label: string; colorR: string; colorT: string; isDashed: boolean }> = []
  
  // 最佳结果
  const bestColor = colorPalette[0]
  items.push({ label: '最佳-初始', colorR: bestColor.R, colorT: bestColor.T, isDashed: true })
  if (enableOptimization.value && optimizedSpectrum.value) {
    items.push({ label: '最佳-优化后', colorR: bestColor.R, colorT: bestColor.T, isDashed: false })
  }
  
  // 选中的采样结果
  const samples = Array.isArray(selectedSamples.value) ? selectedSamples.value : []
  samples.forEach((sampleIdx, colorIdx) => {
    const spectrum = sampleSpectra.value.get(sampleIdx)
    if (!spectrum) return
    
    const color = colorPalette[(colorIdx + 1) % colorPalette.length]
    const label = `#${String(sampleIdx).padStart(2, '0')}`
    
    if (spectrum.initial) {
      items.push({ label: `${label}-初始`, colorR: color.R, colorT: color.T, isDashed: true })
    }
    if (spectrum.optimized) {
      items.push({ label: `${label}-优化后`, colorR: color.R, colorT: color.T, isDashed: false })
    }
  })
  
  legendContainer.value.innerHTML = items.map((item) => {
    const dashStyle = item.isDashed ? 'stroke-dasharray: 5,5;' : ''
    return `
      <div style="display: flex; align-items: center; margin-bottom: 8px; font-size: 11px;">
        <svg width="30" height="2" style="margin-right: 8px;">
          <line x1="0" y1="1" x2="30" y2="1" stroke="${item.colorR}" stroke-width="2" style="${dashStyle}" />
        </svg>
        <span style="color: #000;">R - ${item.label}</span>
      </div>
      <div style="display: flex; align-items: center; margin-bottom: 8px; font-size: 11px;">
        <svg width="30" height="2" style="margin-right: 8px;">
          <line x1="0" y1="1" x2="30" y2="1" stroke="${item.colorT}" stroke-width="2" style="${dashStyle}" />
        </svg>
        <span style="color: #000;">T - ${item.label}</span>
      </div>
    `
  }).join('')
}

function onCanvasMouseMove(event: MouseEvent) {
  if (!spectrumCanvas.value || !bestSpectrum.value) return
  
  const rect = spectrumCanvas.value.getBoundingClientRect()
  const x = (event.clientX - rect.left) * (devicePixelRatio || 1)
  const y = (event.clientY - rect.top) * (devicePixelRatio || 1)
  
  const W = 600
  const H = 400
  const leftPad = 60
  const rightPad = 40
  const topPad = 40
  const bottomPad = 50
  const plotW = W - leftPad - rightPad
  const plotH = H - topPad - bottomPad
  
  const wavelengths = lamNm.value
  const wlMin = wavelengths[0]
  const wlMax = wavelengths[wavelengths.length - 1]
  
  // 计算当前鼠标位置对应的波长索引
  const relX = (x - leftPad) / plotW
  if (relX < 0 || relX > 1) {
    hoveredCurve.value = null
    return
  }
  
  const wavelength = wlMin + relX * (wlMax - wlMin)
  const wlIdx = Math.round((wavelength - wlMin) / ((wlMax - wlMin) / (wavelengths.length - 1)))
  const clampedIdx = Math.max(0, Math.min(wavelengths.length - 1, wlIdx))
  
  // 计算所有曲线在该波长处的Y值，找出最接近鼠标位置的曲线
  let minDist = Infinity
  let closestCurve: string | null = null
  
  // 检查最佳结果的曲线
  if (bestSpectrum.value) {
    const bestR = bestSpectrum.value.R[clampedIdx] * 100
    const bestT = bestSpectrum.value.T[clampedIdx] * 100
    const bestRY = topPad + plotH * (1 - bestR / 100)
    const bestTY = topPad + plotH * (1 - bestT / 100)
    
    const distR = Math.abs(y - bestRY)
    const distT = Math.abs(y - bestTY)
    
    if (distR < minDist && distR < 20) {
      minDist = distR
      closestCurve = '最佳-初始 R'
    }
    if (distT < minDist && distT < 20) {
      minDist = distT
      closestCurve = '最佳-初始 T'
    }
    
    if (enableOptimization.value && optimizedSpectrum.value) {
      const optR = optimizedSpectrum.value.R[clampedIdx] * 100
      const optT = optimizedSpectrum.value.T[clampedIdx] * 100
      const optRY = topPad + plotH * (1 - optR / 100)
      const optTY = topPad + plotH * (1 - optT / 100)
      
      const distOptR = Math.abs(y - optRY)
      const distOptT = Math.abs(y - optTY)
      
      if (distOptR < minDist && distOptR < 20) {
        minDist = distOptR
        closestCurve = '最佳-优化后 R'
      }
      if (distOptT < minDist && distOptT < 20) {
        minDist = distOptT
        closestCurve = '最佳-优化后 T'
      }
    }
  }
  
  // 检查选中采样结果的曲线
  const samples = Array.isArray(selectedSamples.value) ? selectedSamples.value : []
  samples.forEach(sampleIdx => {
    const spectrum = sampleSpectra.value.get(sampleIdx)
    const label = `#${String(sampleIdx).padStart(2, '0')}`
    
    if (spectrum?.initial) {
      const r = spectrum.initial.R[clampedIdx] * 100
      const t = spectrum.initial.T[clampedIdx] * 100
      const rY = topPad + plotH * (1 - r / 100)
      const tY = topPad + plotH * (1 - t / 100)
      
      const distR = Math.abs(y - rY)
      const distT = Math.abs(y - tY)
      
      if (distR < minDist && distR < 20) {
        minDist = distR
        closestCurve = `${label}-初始 R`
      }
      if (distT < minDist && distT < 20) {
        minDist = distT
        closestCurve = `${label}-初始 T`
      }
    }
    
    if (spectrum?.optimized) {
      const r = spectrum.optimized.R[clampedIdx] * 100
      const t = spectrum.optimized.T[clampedIdx] * 100
      const rY = topPad + plotH * (1 - r / 100)
      const tY = topPad + plotH * (1 - t / 100)
      
      const distR = Math.abs(y - rY)
      const distT = Math.abs(y - tY)
      
      if (distR < minDist && distR < 20) {
        minDist = distR
        closestCurve = `${label}-优化后 R`
      }
      if (distT < minDist && distT < 20) {
        minDist = distT
        closestCurve = `${label}-优化后 T`
      }
    }
  })
  
  hoveredCurve.value = closestCurve
}

function onCanvasMouseLeave() {
  hoveredCurve.value = null
}

function drawPlots (): void {
  // RT
  if (rtCanvas.value) {
    const c = rtCanvas.value; const ctx = c.getContext('2d') as CanvasRenderingContext2D
    const W = c.clientWidth; const H = c.clientHeight
    c.width = Math.max(1, Math.floor(W * devicePixelRatio)); c.height = Math.max(1, Math.floor(H * devicePixelRatio));
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(devicePixelRatio, devicePixelRatio)
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H)
    const pad = 24; ctx.strokeStyle = '#e5e7eb'; ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad)
    const xs: number[] = []; const yR: number[] = []; const yT: number[] = []
    for (let i = 0; i < lamNm.value.length; i++) { const x = pad + (i / (lamNm.value.length - 1)) * (W - 2 * pad); xs.push(x); yR.push(pad + (1 - (R_target.value[i] ?? 0)) * (H - 2 * pad)); yT.push(pad + (1 - (T_target.value[i] ?? 0)) * (H - 2 * pad)) }
    ctx.lineWidth = 2; ctx.strokeStyle = '#2563eb'; drawLine(ctx, xs, yR); ctx.strokeStyle = '#10b981'; drawLine(ctx, xs, yT)
    ctx.fillStyle = '#6b7280'; ctx.fillText('R(蓝) / T(绿)', pad + 6, pad + 14)
  }
  // Weights
  if (wCanvas.value) {
    const c = wCanvas.value; const ctx = c.getContext('2d') as CanvasRenderingContext2D
    const W = c.clientWidth; const H = c.clientHeight
    c.width = Math.max(1, Math.floor(W * devicePixelRatio)); c.height = Math.max(1, Math.floor(H * devicePixelRatio));
    ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.scale(devicePixelRatio, devicePixelRatio)
    ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H)
    const pad = 24; ctx.strokeStyle = '#e5e7eb'; ctx.strokeRect(pad, pad, W - 2 * pad, H - 2 * pad)
    const xs: number[] = []; const yWR: number[] = []; const yWT: number[] = []
    const maxW = Math.max(1, ...spec_weights.value)
    const L = lamNm.value.length
    for (let i = 0; i < L; i++) { const x = pad + (i / (L - 1)) * (W - 2 * pad); xs.push(x); const wR = spec_weights.value[i] ?? 0; const wT = spec_weights.value[i + L] ?? 0; yWR.push(pad + (1 - wR / maxW) * (H - 2 * pad)); yWT.push(pad + (1 - wT / maxW) * (H - 2 * pad)) }
    ctx.lineWidth = 2; ctx.strokeStyle = '#ef4444'; drawLine(ctx, xs, yWR); ctx.strokeStyle = '#f59e0b'; drawLine(ctx, xs, yWT)
    ctx.fillStyle = '#6b7280'; ctx.fillText('wR(红) / wT(橙)  (归一到 max(weight))', pad + 6, pad + 14)
  }
}

function safeDraw () {
  requestAnimationFrame(() => { 
    nextTick().then(() => { 
      if (rtCanvas.value && wCanvas.value) drawPlots()
      if (spectrumCanvas.value && bestSpectrum.value) {
        drawSpectrumChart()
        updateLegend()
      }
    }) 
  })
}

onMounted(() => {
  // 从 localStorage 恢复结果
  if (loadResultsFromStorage()) {
    safeDraw()
  }
  recompute()
  safeDraw()
})
// 避免频繁强制重绘，onUpdated 仅做一次安全调度
onUpdated(() => safeDraw())

/* ---------- 导出 / 提交 ---------- */
function payload () {
  const directives_text = rowsToDirectives().trimEnd()  // 保留行分隔

  return {
    lam_nm: lamNm.value,
    directives_text,          // ← 传这个
    R_target: R_target.value,
    T_target: T_target.value,
    spec_weights: spec_weights.value,
    tf_params: {
      N_vec: N_vec.value,
      Tol_vec: Tol_vec.value,
      I_vec: I_vec.value,
      D_vec: D_vec.value,
      k: tfK.value,
    },
    weight_strategy: {
      mode: weightMode.value,
      gaussian: useGaussian.value ? { sigma_nm: sigmaNm.value, base: gaussBase.value } : null,
    },
    top_kp: topKpCount.value, // Top-kp 数量
    rows: rows.value,         // 保留原字段没问题
  }
}
function emitPayload (): void { if (parseError.value) return; (window as any).lastOptoPayload = payload() }
function copyJSON (): void { navigator.clipboard.writeText(JSON.stringify(payload(), null, 2)) }
function downloadJSON (): void { const blob = new Blob([JSON.stringify(payload(), null, 2)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = `opto_targets_${Date.now()}.json`; a.click(); URL.revokeObjectURL(url) }

const backendResp = ref<BackendResp | null>(null)
const sampleTable = ref<SampleItem[]>([])
const bestOne = ref<BestItem | null>(null)
// 最佳膜系的TMM计算结果
const bestSpectrum = ref<{ R: number[]; T: number[] } | null>(null)
const optimizedSpectrum = ref<{ R: number[]; T: number[]; optimized_structure: string[] } | null>(null)
const spectrumCanvas = ref<HTMLCanvasElement | null>(null)
const legendContainer = ref<HTMLElement | null>(null)
const selectedSamples = ref<number[]>([])  // 选中的采样结果索引
const sampleSpectra = ref<Map<number, { 
  initial: { R: number[]; T: number[] } | null; 
  optimized: { R: number[]; T: number[]; optimized_structure: string[] } | null 
}>>(new Map())
const hoveredCurve = ref<string | null>(null)

// 颜色方案
const colorPalette = [
  { R: '#1f77b4', T: '#ff7f0e' },  // 最佳结果
  { R: '#9467bd', T: '#8c564b' },  // #00
  { R: '#2ca02c', T: '#d62728' },  // #01
  { R: '#ff7f0e', T: '#bcbd22' },  // #02
  { R: '#17becf', T: '#e377c2' },  // #03
  { R: '#7f7f7f', T: '#c7c7c7' },  // #04
  { R: '#bcbd22', T: '#17becf' },  // #05
]
const enableOptimization = ref<boolean>(false)

// 监听bestSpectrum变化，确保图表绘制
watch(bestSpectrum, () => {
  if (bestSpectrum.value) {
    nextTick().then(() => {
      setTimeout(() => {
        if (spectrumCanvas.value && bestSpectrum.value) {
          drawSpectrumChart()
        }
      }, 100)
    })
  }
}, { deep: true })


// localStorage 键名
const STORAGE_KEY = 'optogpt_results'

// 保存结果到 localStorage
function saveResultsToStorage() {
  try {
    const data = {
      backendResp: backendResp.value,
      sampleTable: sampleTable.value,
      bestOne: bestOne.value,
      timestamp: Date.now()
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(data))
  } catch (e) {
    console.warn('[Storage] 保存结果失败:', e)
  }
}

// 从 localStorage 恢复结果
function loadResultsFromStorage() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (!saved) return false
    const data = JSON.parse(saved)
    if (data.backendResp) backendResp.value = data.backendResp
    if (data.sampleTable) sampleTable.value = data.sampleTable
    if (data.bestOne) {
      bestOne.value = data.bestOne
      // 恢复时也计算光谱
      if (data.bestOne?.structure && data.bestOne.structure.length > 0) {
        calculateBestSpectrum(data.bestOne.structure)
      }
    }
    return true
  } catch (e) {
    console.warn('[Storage] 加载结果失败:', e)
    return false
  }
}

// 清除保存的结果
function clearSavedResults() {
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch (e) {
    console.warn('[Storage] 清除结果失败:', e)
  }
}

const REQUEST_TIMEOUT_MS = 20000
async function runBackend (): Promise<void> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)

  try {
    const data = payload()

    // 走 Vite 代理：/api -> 8174
    const resp = await fetch('/api/optogpt/infer/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(data),
      signal: controller.signal,
      credentials: 'omit',
    })

    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      throw new Error(`HTTP ${resp.status} ${resp.statusText}${text ? ` — ${text.slice(0, 300)}` : ''}`)
    }

    const ctype = resp.headers.get('content-type') || ''
    const json: BackendResp = ctype.includes('application/json') ? await resp.json() : {}
    backendResp.value = json

    // ========= 归一化 samples =========
    // 1) "最佳"优先从 json.best 读；若无，则从 samples 里找 idx<0 或 tag 包含 'best'
    // 2) 表格只展示 idx>=0 的项，按 idx 升序，并截断到 topKpCount 条
    const raw: SampleItem[] = Array.isArray(json.samples) ? json.samples : []

    let best: BestItem | null = json.best ?? null
    if (!best) {
      const bestInRaw = raw.find(s =>
        typeof s.idx === 'number' && s.idx < 0 || /best/i.test(String(s.tag || ''))
      )
      if (bestInRaw) best = { source: 'Server', score: bestInRaw.tf, structure: bestInRaw.structure }
    }

    const normalized: SampleItem[] = raw
      .filter(s => typeof s.idx === 'number' && s.idx >= 0)
      .sort((a, b) => (a.idx as number) - (b.idx as number))
      .slice(0, topKpCount.value)

    if (normalized.length > 0) {
      sampleTable.value = normalized
      bestOne.value = best
    } else {
      // 兼容“仅返回最优”的老格式
      sampleTable.value = json.structure
        ? [{ idx: 0, tf: json.tf_score ?? Number.NaN, structure: json.structure }]
        : []
      bestOne.value = json.structure
        ? { source: 'Greedy', score: json.tf_score, structure: json.structure }
        : best
    }
    // ========= /归一化 =========

    // 如果有最佳结果，计算其TMM光谱
    if (bestOne.value?.structure && bestOne.value.structure.length > 0) {
      calculateBestSpectrum(bestOne.value.structure)
      // 如果优化开关已打开，执行优化
      if (enableOptimization.value) {
        optimizeBestStructure(bestOne.value.structure)
      }
    } else {
      bestSpectrum.value = null
      optimizedSpectrum.value = null
    }

    // 保存结果到 localStorage（仅在成功时保存）
    saveResultsToStorage()
    
    safeDraw()
  } catch (err: any) {
    let msg = String(err?.message ?? err)
    if (err?.name === 'AbortError') msg = `请求超过 ${REQUEST_TIMEOUT_MS / 1000}s 超时被取消`
    else if (/Failed to fetch|NetworkError/i.test(msg)) msg = '网络连接失败：后端未启动/代理未通或被浏览器拦截'
    console.error('[infer] 请求失败：', err)
    backendResp.value = { error: msg }
    sampleTable.value = []
    bestOne.value = null
    bestSpectrum.value = null
    optimizedSpectrum.value = null
    // 错误时不保存结果，但可以保留之前成功的结果
  } finally {
    clearTimeout(timer)
  }
}
/* ---------- 计算最佳膜系的TMM光谱 ---------- */
async function calculateBestSpectrum(structure: string[]) {
  if (!structure || structure.length === 0) {
    bestSpectrum.value = null
    return
  }
  
  try {
    const resp = await fetch('/api/optogpt/calculate-spectrum/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ structure }),
      credentials: 'omit',
    })
    
    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      console.error('[calculate-spectrum] 请求失败:', resp.status, text)
      bestSpectrum.value = null
      return
    }
    
    const json = await resp.json()
    if (json.ok && json.R && json.T) {
      console.log('[calculate-spectrum] 计算成功，R长度:', json.R.length, 'T长度:', json.T.length)
      bestSpectrum.value = { R: json.R, T: json.T }
      // 等待DOM更新后再绘制
      nextTick().then(() => {
        setTimeout(() => {
          if (spectrumCanvas.value && bestSpectrum.value) {
            console.log('[calculate-spectrum] 开始绘制图表')
            drawSpectrumChart()
            updateLegend()
          } else {
            console.warn('[calculate-spectrum] 画布或数据未就绪:', {
              hasCanvas: !!spectrumCanvas.value,
              hasSpectrum: !!bestSpectrum.value
            })
          }
        }, 200)
      })
    } else {
      console.error('[calculate-spectrum] 返回格式错误:', json)
      bestSpectrum.value = null
    }
  } catch (err: any) {
    console.error('[calculate-spectrum] 计算失败:', err)
    bestSpectrum.value = null
  }
}

/* ---------- 优化最佳膜系 ---------- */
async function optimizeBestStructure(structure: string[]) {
  if (!structure || structure.length === 0) {
    optimizedSpectrum.value = null
    return
  }
  
  try {
    console.log('[optimize] 开始优化膜系结构:', structure)
    const resp = await fetch('/api/optogpt/optimize-structure/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ 
        structure,
        target_R: R_target.value,
        target_T: T_target.value,
        wavelengths: lamNm.value
      }),
      credentials: 'omit',
    })
    
    if (!resp.ok) {
      const text = await resp.text().catch(() => '')
      console.error('[optimize] 请求失败:', resp.status, text)
      optimizedSpectrum.value = null
      return
    }
    
    const json = await resp.json()
    if (json.ok && json.R && json.T && json.optimized_structure) {
      console.log('[optimize] 优化成功')
      optimizedSpectrum.value = { 
        R: json.R, 
        T: json.T,
        optimized_structure: json.optimized_structure
      }
      // 重新绘制图表
      nextTick().then(() => {
        setTimeout(() => {
          if (spectrumCanvas.value && bestSpectrum.value) {
            drawSpectrumChart()
            updateLegend()
          }
        }, 100)
      })
    } else {
      console.error('[optimize] 返回格式错误:', json)
      optimizedSpectrum.value = null
    }
  } catch (err: any) {
    console.error('[optimize] 优化失败:', err)
    optimizedSpectrum.value = null
  }
}

/* ---------- 采样结果选择处理 ---------- */
async function onSampleSelectionChange(sampleIdx: number, checked: boolean) {
  if (!Array.isArray(selectedSamples.value)) {
    selectedSamples.value = []
  }
  
  if (checked) {
    if (!selectedSamples.value.includes(sampleIdx)) {
      selectedSamples.value.push(sampleIdx)
    }
    // 选中：计算该采样结果的光谱
    const sample = sampleTable.value[sampleIdx]
    if (!sample || !sample.structure) return
    
    // 初始化该采样结果的光谱数据
    if (!sampleSpectra.value.has(sampleIdx)) {
      sampleSpectra.value.set(sampleIdx, { initial: null, optimized: null })
    }
    
    // 计算初始光谱
    try {
      const resp = await fetch('/api/optogpt/calculate-spectrum/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
        body: JSON.stringify({ structure: sample.structure }),
        credentials: 'omit',
      })
      
      if (resp.ok) {
        const json = await resp.json()
        if (json.ok && json.R && json.T) {
          const spectrum = sampleSpectra.value.get(sampleIdx)!
          spectrum.initial = { R: json.R, T: json.T }
          
          // 如果优化开关打开，也计算优化后的光谱
          if (enableOptimization.value) {
            await optimizeSampleStructure(sampleIdx, sample.structure)
          }
          
          drawSpectrumChart()
          updateLegend()
        }
      }
    } catch (err: any) {
      console.error(`[sample-${sampleIdx}] 计算光谱失败:`, err)
    }
  } else {
    // 取消选中：从列表中移除
    const index = selectedSamples.value.indexOf(sampleIdx)
    if (index > -1) {
      selectedSamples.value.splice(index, 1)
    }
    // 移除该采样结果的光谱数据
    sampleSpectra.value.delete(sampleIdx)
    drawSpectrumChart()
    updateLegend()
  }
}

async function optimizeSampleStructure(sampleIdx: number, structure: string[]) {
  try {
    const resp = await fetch('/api/optogpt/optimize-structure/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ 
        structure,
        target_R: R_target.value,
        target_T: T_target.value,
        wavelengths: lamNm.value
      }),
      credentials: 'omit',
    })
    
    if (resp.ok) {
      const json = await resp.json()
      if (json.ok && json.R && json.T && json.optimized_structure) {
        const spectrum = sampleSpectra.value.get(sampleIdx)
        if (spectrum) {
          spectrum.optimized = { 
            R: json.R, 
            T: json.T,
            optimized_structure: json.optimized_structure
          }
          drawSpectrumChart()
          updateLegend()
        }
      }
    }
  } catch (err: any) {
    console.error(`[sample-${sampleIdx}] 优化失败:`, err)
  }
}

/* ---------- 获取优化后的结构 ---------- */
function getOptimizedStructure(sampleIdx: number): string | null {
  const spectrum = sampleSpectra.value.get(sampleIdx)
  if (spectrum?.optimized?.optimized_structure) {
    return spectrum.optimized.optimized_structure.join(', ')
  }
  return null
}

// 监听优化开关变化，更新所有选中采样结果的优化状态
watch(enableOptimization, async (newVal) => {
  if (newVal) {
    // 为所有选中的采样结果执行优化
    const samples = Array.isArray(selectedSamples.value) ? selectedSamples.value : []
    for (const sampleIdx of samples) {
      const sample = sampleTable.value[sampleIdx]
      if (sample?.structure) {
        await optimizeSampleStructure(sampleIdx, sample.structure)
      }
    }
    
    // 优化最佳结果
    if (bestOne.value?.structure) {
      optimizeBestStructure(bestOne.value.structure)
    }
  } else {
    // 清除所有优化结果
    optimizedSpectrum.value = null
    const samples = Array.isArray(selectedSamples.value) ? selectedSamples.value : []
    samples.forEach(sampleIdx => {
      const spectrum = sampleSpectra.value.get(sampleIdx)
      if (spectrum) spectrum.optimized = null
    })
    drawSpectrumChart()
    updateLegend()
  }
})

/* ---------- 解析 token -> 层表 ---------- */
function parseTokens(tokens: string[]) { const rows: Array<{ material: string; thickness: number }> = []; for (const tok of tokens) { const s = String(tok); if (!s.includes('_')) continue; const [mat, raw] = s.split('_', 2); const num = (raw || '').replace(/[^0-9.]/g, ''); if (!num) continue; rows.push({ material: mat, thickness: Number(num) }) } return rows }

const layerDlg = ref<{ open: boolean; rows: Array<{material: string; thickness: number}>; tokens: string[]; tf?: number }>({ open: false, rows: [], tokens: [], tf: undefined })
function openLayers (s: { structure: string[]; tf?: number }) { layerDlg.value = { open: true, rows: parseTokens(s.structure || []), tokens: s.structure || [], tf: s.tf } }
function exportCSV () { const rows = layerDlg.value.rows; const csv = ['index,material,thickness_nm', ...rows.map((r, i) => `${i + 1},${r.material},${r.thickness}`)].join(''); const blob = new Blob([csv], { type: 'text/csv' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = 'layers.csv'; a.click(); URL.revokeObjectURL(url) }

/* ---------- Demo ---------- */
function loadDemo (which: 'basic' | 'notch' | 'absorber') {
  if (which === 'basic') { rows.value = [ { id: uid(), ch: 'R', lam: 385, val: 0.42, w: 2 }, { id: uid(), ch: 'T', lam: 550, val: 0.80, w: 4 }, { id: uid(), ch: 'R', lam: 700, val: 0.30, w: 3 }, ]; weightMode.value = 'fullband'; useGaussian.value = false }
  else if (which === 'notch') { rows.value = [ { id: uid(), ch: 'T', lam: 500, val: 1.0, w: 1 }, { id: uid(), ch: 'T', lam: 550, val: 0.05, w: 4 }, { id: uid(), ch: 'T', lam: 600, val: 1.0, w: 1 }, { id: uid(), ch: 'R', lam: 550, val: 0.95, w: 3 }, ]; weightMode.value = 'anchors'; useGaussian.value = true; sigmaNm.value = 12; gaussBase.value = 0 }
  else { rows.value = [ { id: uid(), ch: 'R', lam: 500, val: 0, w: 3 }, { id: uid(), ch: 'T', lam: 500, val: 0, w: 3 }, { id: uid(), ch: 'R', lam: 700, val: 0, w: 3 }, { id: uid(), ch: 'T', lam: 700, val: 0, w: 3 }, ]; weightMode.value = 'fullband'; useGaussian.value = false }
  safeDraw()
}

/* ---------- 小工具 ---------- */
const n4 = (x: number | string) => Number(x).toFixed(4)
const n6 = (x: number | string) => Number(x).toFixed(6)

/* ---------- Tabs ---------- */
const tab = ref<'preview' | 'table' | 'plot'>('plot')
</script>

<style scoped>
.page { max-width: 1280px; margin: 16px auto 48px; padding: 0 12px; }
.title-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.title { font-size: 20px; font-weight: 600; }
.subtitle { color: #6b7280; font-size: 13px; margin-top: 4px; }
.title-actions > * + * { margin-left: 8px; }
.section-title { font-weight: 600; margin-bottom: 6px; }
.table-scroll { max-height: 320px; overflow: auto; border: 1px solid #eee; border-radius: 8px; }
/* 新增：最多显示 5 行（含表头），其余滚动，可按实际行高微调 */
.table-scroll.max5 { max-height: 340px; }
.grid-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.grid-table thead th { background: #fafafa; position: sticky; top: 0; z-index: 1; }
.grid-table th, .grid-table td { padding: 8px 10px; border-bottom: 1px solid #f0f0f0; }
.grid-table tbody tr:last-child td { border-bottom: none; }
.center { text-align: center; }
.line-actions { display: flex; align-items: center; gap: 8px; margin-top: 10px; }
.submit-row { display: flex; align-items: center; gap: 10px; margin-top: 8px; }
.preview { display: grid; gap: 12px; }
.preview-block .block-title { font-weight: 600; margin-bottom: 6px; }
.mono-box { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; background: #f8fafc; border: 1px dashed #e5e7eb; border-radius: 6px; padding: 10px; white-space: pre-wrap; }
.mono-box.small { max-height: 112px; overflow: auto; }
.two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.plot-grid { display: grid; gap: 16px; }
.plot { width: 100%; height: 220px; border: 1px solid #e5e7eb; border-radius: 8px; }
.plot.small { height: 160px; }
.spectrum-canvas { 
  width: 100%; 
  max-width: 600px; 
  height: 400px; 
  border: 1px solid #e5e7eb; 
  border-radius: 8px; 
  background: #ffffff; 
  display: block;
  cursor: crosshair;
}
.spectrum-legend {
  padding: 12px;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  background: #f9fafb;
  max-height: 400px;
  overflow-y: auto;
  min-width: 200px;
}
.block-title {
  font-weight: 600;
  margin-bottom: 8px;
  font-size: 14px;
}
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }
.best-box { margin-top: 12px; padding: 10px; border: 1px dashed #e5e7eb; border-radius: 8px; background: #f9fafb; }
.best-title { font-weight: 700; margin-bottom: 4px; }
.text-muted { color: #6b7280; }
.mt-1 { margin-top: 6px; }
.hidden { display: none; }
.weight-control-row { display: flex; align-items: center; gap: 12px; }
.weight-control-row .va-select { flex: 1; }
.weight-control-row .va-switch { flex-shrink: 0; margin-top: 0; }
</style>
