<template>
  <div
    class="min-w-[240px] rounded-[14px] border px-4 py-3 shadow-[0_16px_40px_rgba(108,79,61,0.18)]"
    :class="
      type === 'error'
        ? 'border-[#d9a7a0] bg-[#fff1ef]'
        : 'border-[#b9ccb0] bg-[#f5fbf1]'
    "
  >
    <p
      class="text-xs font-semibold uppercase tracking-[0.12em]"
      :class="type === 'error' ? 'text-[#9a3f31]' : 'text-[#4d6b3c]'"
    >
      {{ heading }}
    </p>
    <p class="mt-1 text-sm font-semibold text-[#6c4f3d]">
      {{ message }}
    </p>
  </div>
</template>

<script setup>
defineProps({
  message: {
    type: String,
    default: '',
  },
  type: {
    type: String,
    default: 'success',
  },
  heading: {
    type: String,
    default: 'Saved',
  },
})
</script>
