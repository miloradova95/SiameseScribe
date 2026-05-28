<template>
  <aside
    class="rounded-[14px] border border-[#d8cec5] bg-[#fbf8f5] p-5 text-[#6c4f3d] transition"
    :class="disabled ? 'pointer-events-none opacity-40 grayscale' : ''"
  >
    <div class="mb-4 flex items-start gap-3">
      <div
        class="flex h-6 w-6 items-center justify-center rounded-full bg-[#7b5a49] text-xs font-bold text-white"
      >
        2
      </div>

      <div>
        <h3 class="text-[14px] font-semibold">Improve Model Focus</h3>
        <p class="mt-2 text-[12px] leading-snug text-[#8a7669]">
          Draw directly on either patch to correct the highlighted region. The heatmap
          will turn off automatically when you start drawing.
        </p>
      </div>
    </div>

    <div class="mb-3 grid grid-cols-3 gap-2">
      <button class="tool-btn active" :disabled="disabled" @click="$emit('draw')">
        Draw
      </button>

      <button class="tool-btn" :disabled="disabled" @click="$emit('erase')">
        ⌫ Erase my drawing
      </button>

      <button class="tool-btn" :disabled="disabled" @click="$emit('clear')">
        × Clear my drawing
      </button>
    </div>
    <button class="confirm-btn" :disabled="disabled" @click="$emit('confirm')">
      Confirm drawing
    </button>

    <div class="mt-4 rounded-[8px] bg-[#f3ede8] p-3">
      <h4 class="text-[11px] font-bold">What is the heatmap?</h4>
      <p class="mt-1 text-[11px] leading-snug text-[#8a7669]">
        The heatmap shows where the model focused when comparing patches. Red areas = high
        focus. If it highlights irrelevant areas, draw over the region that should matter.
      </p>
    </div>

    <div class="my-4 flex items-center gap-3">
      <div class="h-px flex-1 bg-[#e1d8d0]"></div>
      <span class="text-[11px] text-[#b3a196]">or</span>
      <div class="h-px flex-1 bg-[#e1d8d0]"></div>
    </div>

    <div>
      <p class="text-[12px] font-semibold">
        Confirm the heatmap focused on the right areas.
      </p>

      <button class="focus-btn mt-2" :disabled="disabled" @click="$emit('focus-correct')">
        ✓ Focus looks correct
      </button>
    </div>

    <div class="mt-5 border-t border-[#e1d8d0] pt-4">
      <p class="text-[12px] text-[#a18e82]">Not sure what to focus on?</p>

      <button class="example-link" :disabled="disabled" @click="$emit('example')">
        View an annotation example →
      </button>
    </div>
  </aside>
</template>

<script setup>
defineEmits(["draw", "erase", "clear", "confirm", "focus-correct", "example"]);

defineProps({
  disabled: {
    type: Boolean,
    default: false,
  },
});
</script>

<style scoped>
.tool-btn {
  border: 1px solid #d8cec5;
  border-radius: 8px;
  background: white;
  padding: 7px 6px;
  font-size: 10px;
  color: #7b5a49;
  transition: 0.15s ease;
}

.tool-btn:hover {
  border-color: #7b5a49;
  background: #f0e7e0;
}

.tool-btn.active {
  background: #8a4b3b;
  color: white;
  border-color: #8a4b3b;
}

.confirm-btn {
  width: 100%;
  border-radius: 8px;
  background: #6c4f3d;
  color: white;
  padding: 9px 12px;
  font-size: 12px;
  font-weight: 600;
}

.focus-btn {
  border: 1px solid #d8cec5;
  border-radius: 8px;
  background: white;
  padding: 8px 12px;
  font-size: 11px;
  color: #7b5a49;
}

.example-link {
  margin-top: 6px;
  font-size: 12px;
  color: #b73525;
  font-weight: 600;
  text-decoration: underline;
}
</style>
