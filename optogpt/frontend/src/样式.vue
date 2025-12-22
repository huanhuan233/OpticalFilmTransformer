<template>
    <n-card
      class="opto-card"
      :bordered="false"
    >
      <!-- 头部 -->
      <div class="header">
        <div class="title-block">
          <div class="badge">OptoGPT</div>
          <h2>光学目标与权重输入</h2>
          <p>通道・波长・目标值・权重，一键提交到后端计算</p>
        </div>
        <div class="header-actions">
          <n-space>
            <n-button tertiary size="small" @click="loadDemo('A')">
              填入示例 A
            </n-button>
            <n-button tertiary size="small" @click="loadDemo('B')">
              填入示例 B
            </n-button>
            <n-button type="primary" size="large" @click="submit">
              🚀 提交到后端计算
            </n-button>
          </n-space>
        </div>
      </div>
  
      <!-- 目标行编辑表 -->
      <n-card class="mt-16" size="small" embedded>
        <div class="table-header">
          <span>目标约束（R/T 光谱离散点）</span>
          <n-space>
            <n-button size="small" @click="importCsv">导入 CSV / Excel</n-button>
            <n-button size="small" @click="addRow">新增一行</n-button>
            <n-button size="small" quaternary @click="clearRows">清空</n-button>
          </n-space>
        </div>
  
        <n-data-table
          :columns="columns"
          :data="rows"
          :single-line="false"
          size="small"
        />
      </n-card>
  
      <!-- 底部：参数 + 光谱图 -->
      <div class="bottom">
        <n-card size="small" embedded class="left-panel">
          <n-form label-width="90">
            <n-form-item label="Top-kp 数量">
              <n-select
                v-model:value="topKp"
                :options="topKpOptions"
                style="width: 140px"
              />
            </n-form-item>
  
            <n-form-item label="权重策略">
              <n-radio-group v-model:value="weightStrategy">
                <n-space vertical>
                  <n-radio value="uniform">全谱默认 = 1</n-radio>
                  <n-radio value="gauss">使用高斯扩散权重</n-radio>
                </n-space>
              </n-radio-group>
            </n-form-item>
  
            <n-divider />
  
            <n-form-item label="TOL 基准">
              <n-input-number v-model:value="tolBase" :step="0.01" style="width: 120px" />
            </n-form-item>
  
            <n-form-item label="K 指数">
              <n-input-number v-model:value="kIndex" :step="1" style="width: 120px" />
            </n-form-item>
          </n-form>
        </n-card>
  
        <n-card size="small" embedded class="right-panel">
          <div class="chart-header">
            <div>
              <h3>R / T 光谱（TMM 计算）</h3>
              <p>实时预览当前目标与候选膜系的拟合程度</p>
            </div>
          </div>
          <div id="rt-chart" class="chart-placeholder">
            <!-- 这里挂载 ECharts / Plotly -->
          </div>
        </n-card>
      </div>
    </n-card>
  </template>
  
  <script setup lang="ts">
  import { ref } from 'vue'
  import type { DataTableColumns } from 'naive-ui'
  
  type Row = {
    channel: 'R' | 'T'
    wavelength: number
    target: number
    weight: number
  }
  
  const rows = ref<Row[]>([
    { channel: 'R', wavelength: 385, target: 0.42, weight: 2 },
    { channel: 'T', wavelength: 550, target: 0.8, weight: 4 },
    { channel: 'R', wavelength: 700, target: 0.3, weight: 3 }
  ])
  
  const columns: DataTableColumns<Row> = [
    {
      title: '通道',
      key: 'channel',
      render (row) {
        return row.channel
      }
    },
    {
      title: '波长 (nm)',
      key: 'wavelength'
    },
    {
      title: '目标值',
      key: 'target'
    },
    {
      title: '权重',
      key: 'weight'
    }
  ]
  
  const topKp = ref(20)
  const topKpOptions = [10, 20, 50].map(v => ({ label: String(v), value: v }))
  const weightStrategy = ref<'uniform' | 'gauss'>('uniform')
  const tolBase = ref(0.05)
  const kIndex = ref(2)
  
  const addRow = () => {
    rows.value.push({ channel: 'R', wavelength: 550, target: 0.5, weight: 1 })
  }
  const clearRows = () => { rows.value = [] }
  const importCsv = () => {}
  const loadDemo = (tag: string) => {}
  const submit = () => {}
  </script>
  
  <style scoped>
  .opto-card {
    background: radial-gradient(circle at top left, #1f2933, #020617);
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.6);
    padding: 20px;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
  }
  .title-block h2 {
    font-size: 20px;
    font-weight: 600;
  }
  .title-block p {
    margin-top: 6px;
    font-size: 12px;
    opacity: 0.7;
  }
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    background: linear-gradient(90deg, #22d3ee, #6366f1);
    color: #0b1120;
    margin-bottom: 6px;
  }
  .table-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 13px;
  }
  .bottom {
    display: grid;
    grid-template-columns: 1.1fr 1.6fr;
    gap: 16px;
    margin-top: 18px;
  }
  .chart-header h3 {
    margin: 0;
    font-size: 14px;
  }
  .chart-header p {
    margin: 4px 0 8px;
    font-size: 12px;
    opacity: 0.6;
  }
  .chart-placeholder {
    height: 260px;
    border-radius: 12px;
    background: radial-gradient(circle at top, #020617, #020617);
    border: 1px solid rgba(148, 163, 184, 0.35);
  }
  </style>
  