<template>
  <div class="p-8 max-w-6xl mx-auto space-y-10">
    <section>
      <h1 class="text-2xl font-semibold text-[#2b211d] mb-6">Admin Panel</h1>

      <div class="mb-8 p-4 border border-gray-200 rounded">
        <h2 class="font-semibold mb-3">Create User</h2>
        <form @submit.prevent="createUser" class="flex flex-wrap gap-3 items-end">
          <div>
            <label class="block text-xs mb-1">Username</label>
            <input v-model="form.username" type="text" required class="border rounded px-2 py-1 text-sm" />
          </div>
          <div>
            <label class="block text-xs mb-1">Email</label>
            <input v-model="form.email" type="email" required class="border rounded px-2 py-1 text-sm" />
          </div>
          <div>
            <label class="block text-xs mb-1">Password</label>
            <input v-model="form.password" type="password" required class="border rounded px-2 py-1 text-sm" />
          </div>
          <div>
            <label class="block text-xs mb-1">Role</label>
            <select v-model="form.role" class="border rounded px-2 py-1 text-sm">
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </div>
          <button type="submit" class="bg-[#2b211d] text-white px-4 py-1 rounded text-sm hover:bg-[#c53114] transition">
            Create
          </button>
        </form>
        <p v-if="createError" class="text-red-600 text-sm mt-2">{{ createError }}</p>
        <p v-if="createSuccess" class="text-green-600 text-sm mt-2">{{ createSuccess }}</p>
      </div>

      <table class="w-full text-sm border-collapse">
        <thead>
          <tr class="border-b text-left text-xs uppercase text-gray-500">
            <th class="py-2 pr-4">Username</th>
            <th class="py-2 pr-4">Email</th>
            <th class="py-2 pr-4">Role</th>
            <th class="py-2 pr-4">Active</th>
            <th class="py-2"></th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="u in users" :key="u.id" class="border-b">
            <td class="py-2 pr-4">{{ u.username }}</td>
            <td class="py-2 pr-4">{{ u.email }}</td>
            <td class="py-2 pr-4">{{ u.role }}</td>
            <td class="py-2 pr-4">{{ u.is_active ? 'Yes' : 'No' }}</td>
            <td class="py-2">
              <button
                v-if="u.is_active && u.id !== authStore.user?.id"
                @click="deactivate(u.id)"
                class="text-red-600 text-xs hover:underline"
              >
                Deactivate
              </button>
            </td>
          </tr>
        </tbody>
      </table>
      <p v-if="loadError" class="text-red-600 text-sm mt-4">{{ loadError }}</p>
    </section>

    <section class="border border-gray-200 rounded p-5">
      <div class="flex flex-wrap items-end justify-between gap-4 mb-5">
        <div>
          <h2 class="text-xl font-semibold text-[#2b211d]">Feedback Retraining</h2>
          <p class="text-sm text-gray-600 mt-1">
            Filter feedback, choose the entries you want to train on, and start a backend-managed retrain job.
          </p>
        </div>
        <div class="text-sm text-gray-600">
          Selected: <span class="font-semibold text-[#2b211d]">{{ selectedCount }}</span>
        </div>
      </div>

      <form @submit.prevent="loadAdminFeedback" class="grid gap-4 md:grid-cols-5 mb-5">
        <div>
          <label class="block text-xs mb-1">User</label>
          <select v-model="feedbackFilters.userId" class="w-full border rounded px-2 py-1 text-sm">
            <option value="">All users</option>
            <option v-for="u in users" :key="u.id" :value="String(u.id)">
              {{ u.username }} (#{{ u.id }})
            </option>
          </select>
        </div>
        <div>
          <label class="block text-xs mb-1">From</label>
          <input v-model="feedbackFilters.dateFrom" type="date" class="w-full border rounded px-2 py-1 text-sm" />
        </div>
        <div>
          <label class="block text-xs mb-1">To</label>
          <input v-model="feedbackFilters.dateTo" type="date" class="w-full border rounded px-2 py-1 text-sm" />
        </div>
        <div>
          <label class="block text-xs mb-1">In Retrain</label>
          <select v-model="feedbackFilters.usedForRetrain" class="w-full border rounded px-2 py-1 text-sm">
            <option value="">All</option>
            <option value="false">Available</option>
            <option value="true">Currently used</option>
          </select>
        </div>
        <div>
          <label class="block text-xs mb-1">Triplets / query</label>
          <input
            v-model.number="kTriplets"
            type="number"
            min="1"
            max="5"
            step="1"
            class="w-full border rounded px-2 py-1 text-sm"
          />
        </div>
        <div class="md:col-span-5 flex flex-wrap gap-3">
          <button type="submit" class="bg-[#2b211d] text-white px-4 py-2 rounded text-sm hover:bg-[#c53114] transition">
            Apply Filters
          </button>
          <button type="button" class="border border-gray-300 px-4 py-2 rounded text-sm" @click="resetFeedbackFilters">
            Reset
          </button>
          <button
            type="button"
            class="border border-gray-300 px-4 py-2 rounded text-sm"
            @click="toggleSelectAllVisible"
            :disabled="!feedbackItems.length"
          >
            {{ allVisibleSelected ? 'Clear Visible' : 'Select Visible' }}
          </button>
          <button
            type="button"
            class="bg-[#c53114] text-white px-4 py-2 rounded text-sm disabled:opacity-50"
            @click="startRetrain"
            :disabled="retrainLoading || !selectedCount"
          >
            {{ retrainLoading ? 'Starting...' : 'Start Retrain' }}
          </button>
        </div>
      </form>

      <p v-if="feedbackError" class="text-red-600 text-sm mb-3">{{ feedbackError }}</p>
      <p v-if="retrainError" class="text-red-600 text-sm mb-3">{{ retrainError }}</p>
      <p v-if="retrainSuccess" class="text-green-600 text-sm mb-3">{{ retrainSuccess }}</p>

      <div class="overflow-x-auto">
        <table class="w-full text-sm border-collapse">
          <thead>
            <tr class="border-b text-left text-xs uppercase text-gray-500">
              <th class="py-2 pr-3">Use</th>
              <th class="py-2 pr-3">Date</th>
              <th class="py-2 pr-3">User</th>
              <th class="py-2 pr-3">Label</th>
              <th class="py-2 pr-3">Query Patch</th>
              <th class="py-2 pr-3">Result Patch</th>
              <th class="py-2 pr-3">used_for_retrain</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in feedbackItems" :key="item.id" class="border-b align-top">
              <td class="py-2 pr-3">
                <input
                  :checked="selectedFeedbackIds.has(item.id)"
                  :disabled="item.used_for_retrain"
                  type="checkbox"
                  @change="toggleFeedbackSelection(item.id)"
                />
              </td>
              <td class="py-2 pr-3 whitespace-nowrap">{{ formatFeedbackDate(item.created_at) }}</td>
              <td class="py-2 pr-3">
                <div class="font-medium">{{ item.username }}</div>
                <div class="text-xs text-gray-500">#{{ item.user_id }} • {{ item.user_email }}</div>
              </td>
              <td class="py-2 pr-3">{{ item.label }}</td>
              <td class="py-2 pr-3 break-all">{{ item.query_patch_file_name }}</td>
              <td class="py-2 pr-3 break-all">{{ item.result_patch_file_name }}</td>
              <td class="py-2 pr-3">{{ item.used_for_retrain ? 'true' : 'false' }}</td>
            </tr>
            <tr v-if="!feedbackLoading && !feedbackItems.length">
              <td colspan="7" class="py-4 text-center text-gray-500">No feedback matched the current filters.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { fetchWithAuth, apiUrl } from '../lib/api'
import { fetchAdminFeedback, retrainFromFeedback } from '../services/patch-service'

const authStore = useAuthStore()

const users = ref([])
const loadError = ref('')
const createError = ref('')
const createSuccess = ref('')
const form = ref({ username: '', email: '', password: '', role: 'user' })

const feedbackItems = ref([])
const feedbackLoading = ref(false)
const feedbackError = ref('')
const retrainLoading = ref(false)
const retrainError = ref('')
const retrainSuccess = ref('')
const kTriplets = ref(1)
const selectedFeedbackIds = ref(new Set())
const feedbackFilters = ref({
  userId: '',
  dateFrom: '',
  dateTo: '',
  usedForRetrain: 'false',
})

const selectedCount = computed(() => selectedFeedbackIds.value.size)
const visibleSelectableIds = computed(() => feedbackItems.value.filter((item) => !item.used_for_retrain).map((item) => item.id))
const allVisibleSelected = computed(() => (
  visibleSelectableIds.value.length > 0 &&
  visibleSelectableIds.value.every((id) => selectedFeedbackIds.value.has(id))
))

async function loadUsers() {
  loadError.value = ''
  const res = await fetchWithAuth(apiUrl('/users'))
  if (!res.ok) {
    loadError.value = 'Failed to load users'
    return
  }
  users.value = await res.json()
}

async function createUser() {
  createError.value = ''
  createSuccess.value = ''
  const res = await fetchWithAuth(apiUrl('/users'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(form.value),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    createError.value = err.detail ?? 'Failed to create user'
    return
  }
  createSuccess.value = `User "${form.value.username}" created.`
  form.value = { username: '', email: '', password: '', role: 'user' }
  await loadUsers()
}

async function deactivate(userId) {
  const res = await fetchWithAuth(apiUrl(`/users/${userId}/deactivate`), { method: 'PATCH' })
  if (res.ok) {
    await loadUsers()
  }
}

function buildFeedbackFilters() {
  const filters = {
    userId: feedbackFilters.value.userId || undefined,
    usedForRetrain: feedbackFilters.value.usedForRetrain === '' ? undefined : feedbackFilters.value.usedForRetrain,
  }

  if (feedbackFilters.value.dateFrom) {
    filters.dateFrom = `${feedbackFilters.value.dateFrom}T00:00:00Z`
  }
  if (feedbackFilters.value.dateTo) {
    filters.dateTo = `${feedbackFilters.value.dateTo}T23:59:59Z`
  }

  return filters
}

async function loadAdminFeedback() {
  feedbackLoading.value = true
  feedbackError.value = ''
  retrainSuccess.value = ''
  try {
    feedbackItems.value = await fetchAdminFeedback(buildFeedbackFilters())
    syncSelectionWithVisibleRows()
  } catch (error) {
    feedbackError.value = error.message ?? 'Failed to load feedback'
  } finally {
    feedbackLoading.value = false
  }
}

function toggleFeedbackSelection(feedbackId) {
  const next = new Set(selectedFeedbackIds.value)
  if (next.has(feedbackId)) {
    next.delete(feedbackId)
  } else {
    next.add(feedbackId)
  }
  selectedFeedbackIds.value = next
}

function toggleSelectAllVisible() {
  const next = new Set(selectedFeedbackIds.value)
  if (allVisibleSelected.value) {
    visibleSelectableIds.value.forEach((id) => next.delete(id))
  } else {
    visibleSelectableIds.value.forEach((id) => next.add(id))
  }
  selectedFeedbackIds.value = next
}

function syncSelectionWithVisibleRows() {
  const visibleIds = new Set(feedbackItems.value.map((item) => item.id))
  const next = new Set()
  selectedFeedbackIds.value.forEach((id) => {
    if (visibleIds.has(id)) next.add(id)
  })
  selectedFeedbackIds.value = next
}

function resetFeedbackFilters() {
  feedbackFilters.value = {
    userId: '',
    dateFrom: '',
    dateTo: '',
    usedForRetrain: 'false',
  }
  kTriplets.value = 1
  selectedFeedbackIds.value = new Set()
  loadAdminFeedback()
}

async function startRetrain() {
  retrainLoading.value = true
  retrainError.value = ''
  retrainSuccess.value = ''

  try {
    const feedbackIds = Array.from(selectedFeedbackIds.value)
    const result = await retrainFromFeedback({
      feedbackIds,
      kTriplets: Math.round(Number(kTriplets.value)) || 1,
    })
    retrainSuccess.value = `Retrain started for ${result.feedback_count} feedback entr${result.feedback_count === 1 ? 'y' : 'ies'}. Selected rows are marked while the backend job is running and will be released again afterwards.`
    selectedFeedbackIds.value = new Set()
    await loadAdminFeedback()
  } catch (error) {
    retrainError.value = error.message ?? 'Failed to start retraining'
  } finally {
    retrainLoading.value = false
  }
}

function formatFeedbackDate(value) {
  return new Date(value).toLocaleString()
}

onMounted(async () => {
  await loadUsers()
  await loadAdminFeedback()
})
</script>