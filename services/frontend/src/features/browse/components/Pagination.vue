<template>
  <div
    v-if="totalPages > 1"
    class="mt-10 flex items-center justify-center gap-6 text-[15px]"
  >
  
    <button
      class="px-3 py-1 rounded-full transition hover:bg-[#eee7e2] disabled:opacity-30"
      :disabled="currentPage === 1"
      @click="emit('change', currentPage - 1)"
    >
      ‹
    </button>

    <div class="flex items-center gap-3">
      <button
        v-for="item in visiblePages"
        :key="item.key"
        :disabled="item.type === 'dots'"
        class="px-2 py-1 rounded-full transition"
        :class="pageClass(item)"
        @click="item.type === 'page' && emit('change', item.value)"
      >
        {{ item.label }}
      </button>
    </div>

    <button
      class="px-3 py-1 rounded-full transition hover:bg-[#eee7e2] disabled:opacity-30"
      :disabled="currentPage === totalPages"
      @click="emit('change', currentPage + 1)"
    >
      ›
    </button>

    <form
      class="ml-4 flex items-center gap-2"
      @submit.prevent="submitPageJump"
    >
      <span class="text-sm text-[#8a7568]">Go to</span>

      <input
        v-model.number="pageInput"
        type="number"
        min="1"
        :max="totalPages"
        class="h-9 w-16 rounded-full border border-[#d8ccc3] bg-[#fbf8f5] px-3 text-center text-sm outline-none"
      />

      <button
        type="submit"
        class="rounded-full bg-[#6c4f3d] px-3 py-2 text-sm text-white transition hover:opacity-80"
      >
        Go
      </button>
    </form>
  </div>
</template>

<script setup>
import { computed, ref, watch } from 'vue'

const props = defineProps({
  currentPage: Number,
  totalPages: Number,
})

const emit = defineEmits(['change'])

const pageInput = ref(props.currentPage)

watch(
  () => props.currentPage,
  (page) => {
    pageInput.value = page
  }
)

function submitPageJump() {
  const page = Math.min(
    Math.max(Number(pageInput.value), 1),
    props.totalPages
  )
  emit('change', page)
}

const visiblePages = computed(() => {
  const total = props.totalPages
  const current = props.currentPage
  const pages = []

  if (total <= 7) {
    for (let i = 1; i <= total; i++) {
      pages.push(pageItem(i))
    }
    return pages
  }

  pages.push(pageItem(1))

  if (current <= 5) {
    for (let i = 2; i <= 5; i++) {
      pages.push(pageItem(i))
    }
    pages.push(dotItem('dots-end'))
    pages.push(pageItem(total))
    return pages
  }

  if (current >= total - 4) {
    pages.push(dotItem('dots-start'))
    for (let i = total - 4; i <= total; i++) {
      pages.push(pageItem(i))
    }
    return pages
  }

  pages.push(dotItem('dots-start'))
  pages.push(pageItem(current - 1))
  pages.push(pageItem(current))
  pages.push(pageItem(current + 1))
  pages.push(dotItem('dots-end'))
  pages.push(pageItem(total))

  return pages
})

function pageItem(page) {
  return {
    type: 'page',
    value: page,
    label: page,
    key: page,
  }
}

function dotItem(key) {
  return {
    type: 'dots',
    label: '...',
    key,
  }
}

function pageClass(item) {
  if (item.type === 'dots') {
    return 'cursor-default text-[#a99b91]'
  }

  return item.value === props.currentPage
    ? 'font-semibold text-[#2b211d]'
    : 'text-[#a08c80] hover:text-[#6c4f3d]'
}
</script>