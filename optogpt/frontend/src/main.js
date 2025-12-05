import { createApp } from 'vue'
import App from './App.vue'

// ✅ 新增：引入 Vuestic UI
import { createVuestic } from 'vuestic-ui'
import 'vuestic-ui/css'

// 创建 Vue 应用
const app = createApp(App)

// ✅ 注册 Vuestic UI 插件
app.use(createVuestic({
  config: {
    i18n: {
      messages: {
        en: {}, // 可留空
        zh: {
          search: '搜索',
          clear: '清空',
          selectAll: '全选',
          noOptions: '无选项',
          ok: '确定',
          cancel: '取消',
        },
      },
      currentLanguage: 'zh',
    },
  },
}))


// 挂载
app.mount('#app')
