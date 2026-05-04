<template>
  <main class="min-h-[calc(100vh-72px)] bg-[#f7f1eb] text-[#5b4033]">
    <div class="grid min-h-[calc(100vh-72px)] grid-cols-[285px_1fr]">
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

        <button class="mt-auto px-4 py-3 text-left text-xl hover:opacity-70" @click="logout">
          Logout
        </button>
      </aside>

      <section>
        <div class="border-b border-[#d7cec7] px-12 py-10">
          <h1 class="font-display text-[64px] leading-none">
            {{ pageTitle }}
          </h1>
        </div>

        <div class="px-12 py-9">
          <template v-if="activeTab === 'feedback'">
            <div v-if="feedbackLoading" class="py-8 text-lg text-[#8a7568]">
              Loading your feedback...
            </div>

            <div v-else-if="feedbackError" class="py-8 text-lg text-red-700">
              {{ feedbackError }}
            </div>

            <div v-else-if="feedbackItems.length" class="space-y-4">
              <article
                v-for="item in feedbackItems"
                :key="item.id"
                class="grid grid-cols-[160px_1fr_160px] gap-6 rounded-2xl border border-[#d7cec7] bg-white/60 p-5"
              >
                <img
                  :src="getPatchImageUrl(item.query_patch_id)"
                  :alt="item.query_patch_file_name"
                  class="h-32 w-full rounded-xl bg-[#ebe3db] object-cover"
                />

                <div class="flex flex-col justify-center">
                  <div class="mb-3 flex items-center gap-3">
                    <span
                      class="rounded-full px-3 py-1 text-sm font-semibold"
                      :class="item.label === 'similar' ? 'bg-[#ddebdc] text-[#315b2c]' : 'bg-[#f3dfdc] text-[#8f3a2b]'"
                    >
                      {{ item.label === 'similar' ? 'Similar' : 'Not Similar' }}
                    </span>
                    <span class="text-sm text-[#8a7568]">
                      {{ formatFeedbackDate(item.created_at) }}
                    </span>
                  </div>

                  <p class="text-lg font-semibold text-[#5b4033]">
                    {{ item.query_patch_file_name }}
                  </p>
                  <p class="my-1 text-sm uppercase tracking-[0.18em] text-[#b19382]">
                    compared with
                  </p>
                  <p class="text-lg font-semibold text-[#5b4033]">
                    {{ item.result_patch_file_name }}
                  </p>
                </div>

                <img
                  :src="getPatchImageUrl(item.result_patch_id)"
                  :alt="item.result_patch_file_name"
                  class="h-32 w-full rounded-xl bg-[#ebe3db] object-cover"
                />
              </article>
            </div>

            <p v-else class="text-lg text-[#8a7568]">
              You have not saved any feedback yet.
            </p>
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
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import ImageCard from '@/components/ImageCard.vue'
import { apiUrl } from '@/lib/api'
import { fetchMyFeedback } from '@/services/patch-service'

const router = useRouter()
const authStore = useAuthStore()

const activeTab = ref('feedback')
const bookmarks = ref([])
const feedbackItems = ref([])
const feedbackLoading = ref(false)
const feedbackError = ref(null)

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

function getPatchImageUrl(patchId) {
  return apiUrl(`/patches/${patchId}/file`)
}

function formatFeedbackDate(value) {
  return new Date(value).toLocaleString()
}

async function loadFeedback() {
  feedbackLoading.value = true
  feedbackError.value = null

  try {
    feedbackItems.value = await fetchMyFeedback()
  } catch (error) {
    feedbackError.value = error.message || 'Failed to load your feedback.'
  } finally {
    feedbackLoading.value = false
  }
}

function logout() {
  authStore.logout()
  router.push('/login')
}

onMounted(() => {
  loadBookmarks()
  loadFeedback()

  window.addEventListener('bookmarks-updated', loadBookmarks)
  window.addEventListener('storage', loadBookmarks)
})

onBeforeUnmount(() => {
  window.removeEventListener('bookmarks-updated', loadBookmarks)
  window.removeEventListener('storage', loadBookmarks)
})

watch(activeTab, (tab) => {
  if (tab === 'feedback' && !feedbackLoading.value) {
    loadFeedback()
  }
})
</script>
