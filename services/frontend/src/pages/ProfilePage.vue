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

          <button
            v-if="authStore.isAdmin"
            class="w-full rounded-xl px-4 py-3 text-left"
            :class="activeTab === 'admin' ? 'bg-[#d8cdc6]' : 'hover:bg-[#eee7e2]'"
            @click="activeTab = 'admin'"
          >
            Admin Panel
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
            <div class="mb-8 rounded-[28px] border border-[#d7cec7] bg-white/55 px-6 py-6">
              <p class="text-xs uppercase tracking-[0.22em] text-[#b19382]">
                Personal archive
              </p>
              <div class="mt-3 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
                <div>
                  <h2 class="text-3xl font-semibold text-[#5b4033]">
                    Your uploaded manuscript images
                  </h2>
                  <p class="mt-2 max-w-2xl text-base text-[#8a7568]">
                    Images uploaded while signed in are collected here so you can jump back into browsing and annotation quickly.
                  </p>
                </div>

                <div class="rounded-2xl border border-[#dbcfc7] bg-[#f7f1eb] px-5 py-4 text-right">
                  <p class="text-xs uppercase tracking-[0.18em] text-[#b19382]">
                    Total uploads
                  </p>
                  <p class="mt-1 text-3xl font-semibold text-[#5b4033]">
                    {{ myUploads.length }}
                  </p>
                </div>
              </div>
            </div>

            <div v-if="uploadsLoading" class="py-8 text-lg text-[#8a7568]">
              Loading your uploads...
            </div>

            <div v-else-if="uploadsError" class="py-8 text-lg text-red-700">
              {{ uploadsError }}
            </div>

            <div
              v-else-if="myUploads.length"
              class="grid grid-cols-2 gap-8 md:grid-cols-3 xl:grid-cols-4"
            >
              <ImageCard
                v-for="image in myUploads"
                :key="image.id"
                :image="image"
                @click="goToImage(image)"
              />
            </div>

            <div
              v-else
              class="rounded-[28px] border border-dashed border-[#d7cec7] bg-white/30 px-8 py-12 text-center"
            >
              <p class="text-2xl font-semibold text-[#5b4033]">
                No uploads yet
              </p>
              <p class="mt-3 text-lg text-[#8a7568]">
                Upload an image and it will appear here automatically.
              </p>
              <button
                class="mt-6 rounded-full border border-[#5b4033] px-6 py-3 text-base font-medium text-[#5b4033] transition hover:bg-[#eee7e2]"
                @click="router.push('/upload')"
              >
                Go to Upload
              </button>
            </div>
          </template>

          <template v-else-if="activeTab === 'bookmarks'">
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

          <template v-else-if="activeTab === 'admin' && authStore.isAdmin">
            <AdminPanel />
          </template>
        </div>
      </section>
    </div>
  </main>
</template>

<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import ImageCard from '@/components/ImageCard.vue'
import AdminPanel from '@/features/admin/AdminPanel.vue'
import { apiUrl } from '@/lib/api'
import { formatLocalDateTime } from '@/lib/date'
import { clearBookmarkCache, fetchBookmarks } from '@/services/bookmark-service'
import { fetchMyFeedback } from '@/services/patch-service'
import { fetchMyImages } from '@/services/image-service'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

const activeTab = ref('feedback')
const bookmarks = ref([])
const feedbackItems = ref([])
const feedbackLoading = ref(false)
const feedbackError = ref(null)
const myUploads = ref([])
const uploadsLoading = ref(false)
const uploadsError = ref(null)

const pageTitle = computed(() => {
  if (activeTab.value === 'admin') return 'Admin Panel'
  if (activeTab.value === 'bookmarks') return 'Bookmarks'
  if (activeTab.value === 'uploads') return 'My Uploads'
  return 'My Feedback'
})

const bookmarkedImages = computed(() => bookmarks.value)

const allowedTabs = computed(() => {
  const baseTabs = ['feedback', 'uploads', 'bookmarks']
  return authStore.isAdmin ? [...baseTabs, 'admin'] : baseTabs
})

function resolveTab(tab) {
  return allowedTabs.value.includes(tab) ? tab : 'feedback'
}

async function loadBookmarks(force = false) {
  bookmarks.value = await fetchBookmarks(force)
}

function goToImage(image) {
  router.push(`/browse/${encodeURIComponent(image.fileName)}`)
}

function getPatchImageUrl(patchId) {
  return apiUrl(`/patches/${patchId}/file`)
}

function formatFeedbackDate(value) {
  return formatLocalDateTime(value)
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

async function loadUploads() {
  uploadsLoading.value = true
  uploadsError.value = null

  try {
    const images = await fetchMyImages()
    myUploads.value = Array.isArray(images) ? images : []
  } catch (error) {
    uploadsError.value = error.message || 'Failed to load your uploads.'
  } finally {
    uploadsLoading.value = false
  }
}

function logout() {
  authStore.logout()
  clearBookmarkCache()
  router.push('/login')
}

onMounted(() => {
  activeTab.value = resolveTab(route.query.tab)
  loadBookmarks()
  loadFeedback()
  loadUploads()

  window.addEventListener('bookmarks-updated', handleBookmarksUpdated)
})

onBeforeUnmount(() => {
  window.removeEventListener('bookmarks-updated', handleBookmarksUpdated)
})

function handleBookmarksUpdated(event) {
  const updatedBookmarks = event.detail?.bookmarks
  bookmarks.value = Array.isArray(updatedBookmarks) ? updatedBookmarks : []
}

watch(activeTab, (tab) => {
  if (route.query.tab !== tab) {
    router.replace({ query: { ...route.query, tab } })
  }

  if (tab === 'feedback' && !feedbackLoading.value) {
    loadFeedback()
  }

  if (tab === 'uploads' && !uploadsLoading.value) {
    loadUploads()
  }
})

watch(
  () => route.query.tab,
  (tab) => {
    const nextTab = resolveTab(tab)
    if (activeTab.value !== nextTab) {
      activeTab.value = nextTab
    }
  }
)

watch(
  () => authStore.isAdmin,
  (isAdmin) => {
    if (!isAdmin && activeTab.value === 'admin') {
      activeTab.value = 'feedback'
    }
  }
)

watch(
  () => authStore.user?.id,
  () => {
    clearBookmarkCache()
    loadBookmarks(true)
  },
  { immediate: true }
)
</script>
