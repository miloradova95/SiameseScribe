<template>
  <button
    type="button"
    class="flex h-9 w-9 items-center justify-center rounded-full bg-white/80 shadow transition hover:scale-110 hover:bg-white"
    :disabled="bookmarkSaving"
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
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { addBookmark, fetchBookmarks, removeBookmark } from '@/services/bookmark-service'

const props = defineProps({
  item: {
    type: Object,
    required: true,
  },
})

const bookmarks = ref([])
const bookmarkSaving = ref(false)

async function readBookmarks(force = false) {
  bookmarks.value = await fetchBookmarks(force)
}

const isBookmarked = computed(() => {
  return bookmarks.value.some((item) => item.id === props.item.id)
})

async function toggleBookmark() {
  if (bookmarkSaving.value) return

  bookmarkSaving.value = true

  try {
    if (isBookmarked.value) {
      bookmarks.value = await removeBookmark(props.item.id)
      window.dispatchEvent(
        new CustomEvent('app-notification', {
          detail: {
            heading: 'Removed',
            message: 'Bookmark removed.',
            type: 'success',
          },
        }),
      )
    } else {
      bookmarks.value = await addBookmark(props.item.id)
      window.dispatchEvent(
        new CustomEvent('app-notification', {
          detail: {
            heading: 'Saved',
            message: 'Image bookmarked.',
            type: 'success',
          },
        }),
      )
    }
  } catch (error) {
    window.dispatchEvent(
      new CustomEvent('app-notification', {
        detail: {
          heading: 'Notification',
          message: error.message || 'Failed to update bookmark.',
          type: 'error',
        },
      }),
    )
  } finally {
    bookmarkSaving.value = false
  }
}

onMounted(() => {
  readBookmarks()

  window.addEventListener('bookmarks-updated', handleBookmarksUpdated)
})

onBeforeUnmount(() => {
  window.removeEventListener('bookmarks-updated', handleBookmarksUpdated)
})

function handleBookmarksUpdated(event) {
  const updatedBookmarks = event.detail?.bookmarks
  bookmarks.value = Array.isArray(updatedBookmarks) ? updatedBookmarks : []
}

watch(
  () => props.item?.id,
  () => readBookmarks(),
)
</script>
