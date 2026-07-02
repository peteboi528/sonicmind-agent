<script setup>
import { computed } from "vue";

const props = defineProps({
  energy: { type: Number, default: null },     // 0..1 (estimated or measured)
  source: { type: String, default: null },       // 'estimated' | 'measured' | null
});

// 三档：贴合观测区间（约 0.22–0.88）按 0.5 居中切分。
const band = computed(() => {
  const e = props.energy;
  if (e == null) return null;
  if (e < 0.38) return { key: "low", label: "低能量", bars: 1 };
  if (e < 0.62) return { key: "mid", label: "中能量", bars: 2 };
  return { key: "high", label: "高能量", bars: 3 };
});

const estimated = computed(() => props.source === "estimated");
const tooltip = computed(() => {
  if (!band.value) return "";
  const base = `${band.value.label}（能量 ≈ ${(props.energy ?? 0).toFixed(2)}）`;
  if (props.source === "estimated") return `${base} · 基于曲风/情绪标签估算，非真实测量`;
  if (props.source === "measured") return `${base} · 真实音频测量`;
  return base;
});
</script>

<template>
  <span v-if="band" class="energy-badge" :class="band.key" :title="tooltip">
    <span class="bars" aria-hidden="true">
      <i v-for="n in 3" :key="n" class="bar" :class="{ on: n <= band.bars }"></i>
    </span>
    <span class="eb-label">{{ band.label }}</span>
    <span v-if="estimated" class="eb-mark" title="估算值（非真实测量）">估</span>
  </span>
</template>

<style scoped>
.energy-badge {
  display: inline-flex; align-items: center; gap: 5px;
  margin-left: 6px; padding: 2px 8px 2px 7px;
  border-radius: var(--radius-pill);
  border: 1px solid transparent;
  font-size: 0.68rem; font-weight: 600;
  font-family: var(--font-display);
  vertical-align: middle;
  line-height: 1.5;
}

/* 三档配色（暗色 UI 下的低饱和冷暖）：低=冷静蓝，中=琥珀，高=暖橙红 */
.energy-badge.low  { background: rgba(91,141,239,0.14);  border-color: rgba(91,141,239,0.25);  color: #9cbbf7; }
.energy-badge.mid  { background: rgba(240,180,41,0.14);  border-color: rgba(240,180,41,0.25);  color: #f0c45a; }
.energy-badge.high { background: rgba(239,108,77,0.16);  border-color: rgba(239,108,77,0.30);  color: #f2a07a; }

/* 三根竖条：按档位点亮 1/2/3 根，一眼读出能量高低 */
.bars { display: inline-flex; align-items: flex-end; gap: 2px; height: 11px; }
.bar {
  width: 3px; border-radius: 1.5px;
  background: currentColor; opacity: 0.22;
}
.bar:nth-child(1) { height: 5px; }
.bar:nth-child(2) { height: 8px; }
.bar:nth-child(3) { height: 11px; }
.bar.on { opacity: 0.95; }

.eb-mark {
  font-size: 0.6rem; font-weight: 700;
  opacity: 0.62;
  border-bottom: 1px dotted currentColor;
  margin-left: 1px;
}
</style>
