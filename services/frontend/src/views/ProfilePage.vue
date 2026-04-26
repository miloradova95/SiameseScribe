<template>
  <main class="mx-auto max-w-[1860px] border border-[#9f9893] bg-[#fbf8f5]">
    <div class="grid min-h-[900px] grid-cols-[330px_1fr]">
      <aside class="flex flex-col border-r border-[#d7cec7] px-8 py-7">
        <h2 class="mb-6 text-xl font-semibold">Profile</h2>

        <nav class="space-y-2 text-xl">
          <button
            class="w-full rounded-xl px-4 py-3 text-left"
            :class="activeTab === 'feedback' ? 'bg-[#d8cdc6]' : 'hover:bg-[#eee7e2]'"
            @click="activeTab = 'feedback'"
          >
            My Feedback
          </button>

          <button
            class="w-full rounded-xl px-4 py-3 text-left"
            :class="activeTab === 'uploads' ? 'bg-[#d8cdc6]' : 'hover:bg-[#eee7e2]'"
            @click="activeTab = 'uploads'"
          >
            My Uploads
          </button>

          <button
            class="w-full rounded-xl px-4 py-3 text-left"
            :class="activeTab === 'bookmarks' ? 'bg-[#d8cdc6]' : 'hover:bg-[#eee7e2]'"
            @click="activeTab = 'bookmarks'"
          >
            Bookmarks
          </button>
        </nav>

        <button class="mt-auto px-4 py-3 text-left text-xl hover:opacity-70">
          Logout
        </button>
      </aside>

      <section>
        <div class="border-b border-[#d7cec7] px-12 py-10">
          <h1 class="font-serif text-[64px] leading-none">
            {{ pageTitle }}
          </h1>
        </div>

        <div class="px-12 py-9">
          <template v-if="activeTab === 'feedback'">
            <p class="text-lg">My Feedback content here.</p>
          </template>

          <template v-else-if="activeTab === 'uploads'">
            <p class="text-lg">My Uploads content here.</p>
          </template>

          <template v-else>
            <div
              v-if="bookmarkedImages.length"
              class="grid grid-cols-2 gap-8 md:grid-cols-3 xl:grid-cols-4"
            >
              <ImageCard
                v-for="image in bookmarkedImages"
                :key="image.id"
                :image="image"
                @click="goToImage(image)"
              />
            </div>

            <p v-else class="text-lg text-[#8a7568]">
              No bookmarked images yet.
            </p>
          </template>
        </div>
      </section>
    </div>
  </main>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import ImageCard from '@/components/ImageCard.vue'

const router = useRouter()

const activeTab = ref('feedback')
const bookmarks = ref([])

const pageTitle = computed(() => {
  if (activeTab.value === 'bookmarks') return 'Bookmarks'
  if (activeTab.value === 'uploads') return 'My Uploads'
  return 'My Feedback'
})

const bookmarkedImages = computed(() => bookmarks.value)

function loadBookmarks() {
  bookmarks.value = JSON.parse(localStorage.getItem('bookmarks') || '[]')
}

function goToImage(image) {
  router.push(`/browse/${encodeURIComponent(image.fileName)}`)
}

onMounted(() => {
  loadBookmarks()

  window.addEventListener('bookmarks-updated', loadBookmarks)
  window.addEventListener('storage', loadBookmarks)
})
</script>