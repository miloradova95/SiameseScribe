<template>
  <button
    type="button"
    class="flex h-9 w-9 items-center justify-center rounded-full bg-white/80 shadow transition hover:scale-110 hover:bg-white"
    @click.stop="toggleBookmark"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      class="h-6 w-6 transition"
      :class="isBookmarked ? 'text-[#6c4f3d]' : 'text-[#b8aaa0]'"
      stroke="currentColor"
      stroke-width="2"
    >
      <path
        stroke-linecap="round"
        stroke-linejoin="round"
        :fill="isBookmarked ? 'currentColor' : 'none'"
        d="M6 4.75A1.75 1.75 0 0 1 7.75 3h8.5A1.75 1.75 0 0 1 18 4.75V21l-6-3.75L6 21V4.75Z"
      />
    </svg>
  </button>
</template>

<script setup>
import { computed, ref, onMounted, watch } from 'vue'

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
})

const STORAGE_KEY = 'bookmarks'
const bookmarks = ref([])

function readBookmarks() {
  bookmarks.value = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
}

const isBookmarked = computed(() => {
  return bookmarks.value.some((item) => item.id === props.item.id)
})

function toggleBookmark() {
  readBookmarks()

  if (isBookmarked.value) {
    bookmarks.value = bookmarks.value.filter((item) => item.id !== props.item.id)
  } else {
    bookmarks.value.push(props.item)
  }

  localStorage.setItem(STORAGE_KEY, JSON.stringify(bookmarks.value))

  window.dispatchEvent(new Event('bookmarks-updated'))
}

onMounted(() => {
  readBookmarks()

  window.addEventListener('bookmarks-updated', readBookmarks)
  window.addEventListener('storage', readBookmarks)
})

watch(
  () => props.item?.id,
  () => readBookmarks()
)
</script>