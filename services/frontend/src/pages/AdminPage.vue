<template>
  <div class="min-h-screen bg-[#fbfaf8] px-6 py-8 text-[#5b4034]">
    <div class="mx-auto max-w-6xl">
      <div class="mb-8 flex items-start justify-between gap-4">
        <h1 class="text-6xl font-bold tracking-tight text-[#5b4034]">
          Admin
        </h1>

        <button
          type="button"
          class="flex items-center gap-2 rounded-full bg-[#5b4034] px-5 py-2 text-sm text-white transition hover:bg-[#c53114]"
          @click="showCreateUser = true"
        >
          <span class="text-xl leading-none">+</span>
          Create Member
        </button>
      </div>

      <div class="mb-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div class="rounded-xl bg-[#ead8b9] p-5">
          <div class="text-2xl font-bold">{{ activeMembersCount }}</div>
          <div class="text-sm text-[#8a756b]">Active Members</div>
        </div>

        <div class="rounded-xl bg-[#e5d8d1] p-5">
          <div class="text-2xl font-bold">{{ deactivatedMembersCount }}</div>
          <div class="text-sm text-[#8a756b]">Deactivated</div>
        </div>

        <div class="rounded-xl bg-[#e5d8d1] p-5">
          <div class="text-2xl font-bold">{{ users.length }}</div>
          <div class="text-sm text-[#8a756b]">Total Members</div>
        </div>
      </div>

      <div class="mb-4 flex gap-8 border-b border-[#ded2ca]">
        <button
          type="button"
          class="pb-3 text-sm font-semibold"
          :class="activeTab === 'members'
            ? 'border-b-2 border-[#5b4034] text-[#5b4034]'
            : 'text-[#9a867c]'"
          @click="activeTab = 'members'"
        >
          Members
        </button>

        <button
          type="button"
          class="pb-3 text-sm font-semibold"
          :class="activeTab === 'feedback'
            ? 'border-b-2 border-[#5b4034] text-[#5b4034]'
            : 'text-[#9a867c]'"
          @click="activeTab = 'feedback'"
        >
          Feedback Retraining
          <span class="ml-1 rounded-full bg-[#ead8b9] px-2 py-0.5 text-xs text-[#a7441d]">
            {{ feedbackItems.length }}
          </span>
        </button>
      </div>

      <section v-if="activeTab === 'members'">
        <div class="grid grid-cols-[1.5fr_0.8fr_0.8fr_1.2fr] border-b border-[#e5d8d1] px-3 py-4 text-xs font-semibold text-[#9a867c]">
          <div>User</div>
          <div>Status</div>
          <div>Role</div>
          <div></div>
        </div>

        <div
      
<div
  v-for="u in users"
  :key="u.id"
  class="grid grid-cols-[1.5fr_0.8fr_0.8fr_1.2fr] items-center border-b px-3 py-5 transition"
  :class="u.id === authStore.user?.id
    ? 'bg-[#fff7ed] border-[#e3b5a8]'
    : 'border-[#e3b5a8]'"
>
        
          <div class="flex items-center gap-4">
            <div class="flex h-10 w-10 items-center justify-center rounded-full bg-[#e7f2ff] text-sm font-bold text-[#1976d2]">
              {{ u.username?.slice(0, 2).toUpperCase() }}
            </div>

            <div>
              
              <div class="font-semibold text-[#5b4034]">{{ u.username }}</div>
              <div class="text-sm text-[#9a867c]">{{ u.email }}</div>
            </div>
          </div>

          <div>
<span
  class="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold"
  :class="u.is_active
    ? 'bg-[#dff5e3] text-[#2e7d32] ring-1 ring-[#2e7d32]/20'
    : 'bg-[#f1e8e2] text-[#9a867c]'"
>
  <span
    class="h-2 w-2 rounded-full"
    :class="u.is_active ? 'bg-[#2e7d32]' : 'bg-[#9a867c]'"
  ></span>

  {{ u.is_active ? 'Active' : 'Deactivated' }}
</span>
          </div>

          <div class="text-sm text-[#5b4034]">{{ u.role }}</div>

          <div class="flex gap-4">
            <button
              v-if="u.is_active && u.id !== authStore.user?.id"
              @click="deactivate(u.id)"
              class="rounded-full border border-[#bba79d] px-4 py-1.5 text-sm text-[#5b4034] hover:bg-[#f1e8e2]"
            >
              Deactivate
            </button>

            <button
              v-else-if="!u.is_active"
              @click="activate(u.id)"
              class="rounded-full border border-[#91b875] px-4 py-1.5 text-sm text-[#3f7b22] hover:bg-[#e3f1d8]"
            >
              Activate
            </button>

            <span
              v-else
              class="rounded-full border border-[#91b875] px-4 py-1.5 text-sm text-[#3f7b22]"
            >
              Current User
            </span>

              <button
    v-if="u.id !== authStore.user?.id"
    @click="deleteUser(u.id)"
    class="rounded-full border border-[#e3b5a8] px-4 py-1.5 text-sm text-[#c53114] hover:bg-[#f8e4df]"
  >
    Delete
  </button>
            
          </div>
        </div>

        <p v-if="loadError" class="mt-4 text-sm text-red-600">{{ loadError }}</p>
        <p v-if="createError" class="mt-4 text-sm text-red-600">{{ createError }}</p>
        <p v-if="createSuccess" class="mt-4 text-sm text-green-600">{{ createSuccess }}</p>
      </section>

      <section v-if="activeTab === 'feedback'" class="space-y-6">
        <form
          @submit.prevent="loadAdminFeedback"
          class="grid gap-4 rounded-2xl bg-white/70 p-5 md:grid-cols-5"
        >
          <select
            v-model="feedbackFilters.userId"
            class="rounded-lg border border-[#d8c9c0] px-3 py-2 text-sm"
          >
            <option value="">All users</option>
            <option v-for="u in users" :key="u.id" :value="String(u.id)">
              {{ u.username }} (#{{ u.id }})
            </option>
          </select>

          <input
            v-model="feedbackFilters.dateFrom"
            type="date"
            class="rounded-lg border border-[#d8c9c0] px-3 py-2 text-sm"
          />

          <input
            v-model="feedbackFilters.dateTo"
            type="date"
            class="rounded-lg border border-[#d8c9c0] px-3 py-2 text-sm"
          />

          <select
            v-model="feedbackFilters.usedForRetrain"
            class="rounded-lg border border-[#d8c9c0] px-3 py-2 text-sm"
          >
            <option value="">All</option>
            <option value="false">Available</option>
            <option value="true">Currently used</option>
          </select>

          <input
            v-model.number="kTriplets"
            type="number"
            min="1"
            max="5"
            step="1"
            class="rounded-lg border border-[#d8c9c0] px-3 py-2 text-sm"
          />

          <div class="md:col-span-5 flex flex-wrap gap-3">
            <button
              type="submit"
              class="rounded-full bg-[#5b4034] px-5 py-2 text-sm text-white hover:bg-[#c53114]"
            >
              Apply Filters
            </button>

            <button
              type="button"
              class="rounded-full border border-[#bba79d] px-5 py-2 text-sm"
              @click="resetFeedbackFilters"
            >
              Reset
            </button>

            <button
              type="button"
              class="rounded-full border border-[#bba79d] px-5 py-2 text-sm disabled:opacity-40"
              @click="toggleSelectAllVisible"
              :disabled="!feedbackItems.length"
            >
              {{ allVisibleSelected ? 'Clear Visible' : 'Select Visible' }}
            </button>

            <button
              type="button"
              class="rounded-full bg-[#c53114] px-5 py-2 text-sm text-white disabled:opacity-40"
              @click="startRetrain"
              :disabled="retrainLoading || !selectedCount"
            >
              {{ retrainLoading ? 'Starting...' : 'Start Retrain' }}
            </button>
          </div>
        </form>

        <div class="flex justify-end text-sm text-[#8a756b]">
          Selected:
          <span class="ml-1 font-semibold text-[#5b4034]">{{ selectedCount }}</span>
        </div>

        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-[#ded2ca] text-left text-xs text-[#9a867c]">
                <th class="px-3 py-4">Use</th>
                <th class="px-3 py-4">Date</th>
                <th class="px-3 py-4">User</th>
                <th class="px-3 py-4">Label</th>
                <th class="px-3 py-4">Query Patch</th>
                <th class="px-3 py-4">Result Patch</th>
                <th class="px-3 py-4">Used</th>
              </tr>
            </thead>

            <tbody>
              <tr
                v-for="item in feedbackItems"
                :key="item.id"
                class="border-b border-[#eee5df] align-top"
              >
                <td class="px-3 py-4">
                  <input
                    :checked="selectedFeedbackIds.has(item.id)"
                    :disabled="item.used_for_retrain"
                    type="checkbox"
                    @change="toggleFeedbackSelection(item.id)"
                  />
                </td>

                <td class="whitespace-nowrap px-3 py-4">
                  {{ formatFeedbackDate(item.created_at) }}
                </td>

                <td class="px-3 py-4">
                  <div class="font-semibold text-[#5b4034]">{{ item.username }}</div>
                  <div class="text-xs text-[#9a867c]">
                    #{{ item.user_id }} • {{ item.user_email }}
                  </div>
                </td>

                <td class="px-3 py-4">{{ item.label }}</td>
                <td class="break-all px-3 py-4">{{ item.query_patch_file_name }}</td>
                <td class="break-all px-3 py-4">{{ item.result_patch_file_name }}</td>
                <td class="px-3 py-4">{{ item.used_for_retrain ? 'true' : 'false' }}</td>
              </tr>

              <tr v-if="!feedbackLoading && !feedbackItems.length">
                <td colspan="7" class="py-8 text-center text-[#9a867c]">
                  No feedback matched the current filters.
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        <p v-if="feedbackError" class="text-sm text-red-600">{{ feedbackError }}</p>
        <p v-if="retrainError" class="text-sm text-red-600">{{ retrainError }}</p>
        <p v-if="retrainSuccess" class="text-sm text-green-600">{{ retrainSuccess }}</p>
      </section>
    </div>

    <CreateUserModal
      :open="showCreateUser"
      @close="showCreateUser = false"
      @create="handleCreateUser"
    />
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { fetchWithAuth, apiUrl } from '../lib/api'
import { formatLocalDateTime, localDateEndToUtcIso, localDateStartToUtcIso } from '../lib/date'
import { fetchAdminFeedback, retrainFromFeedback } from '../services/patch-service'
import CreateUserModal from '../features/admin/CreateUserModal.vue'

const activeTab = ref('members')
const showCreateUser = ref(false)

const authStore = useAuthStore()

const users = ref([])
const loadError = ref('')
const createError = ref('')
const createSuccess = ref('')

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

const activeMembersCount = computed(() => users.value.filter((u) => u.is_active).length)
const deactivatedMembersCount = computed(() => users.value.filter((u) => !u.is_active).length)

const selectedCount = computed(() => selectedFeedbackIds.value.size)

const visibleSelectableIds = computed(() =>
  feedbackItems.value
    .filter((item) => !item.used_for_retrain)
    .map((item) => item.id)
)

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

async function handleCreateUser(payload) {
  createError.value = ''
  createSuccess.value = ''

  const res = await fetchWithAuth(apiUrl('/users'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    createError.value = err.detail ?? 'Failed to create user'
    return
  }

  createSuccess.value = `User "${payload.username}" created.`
  showCreateUser.value = false
  await loadUsers()
}

async function deactivate(userId) {
  const res = await fetchWithAuth(apiUrl(`/users/${userId}/deactivate`), {
    method: 'PATCH',
  })

  if (res.ok) {
    await loadUsers()
  }
}

async function activate(userId) {
  const res = await fetchWithAuth(apiUrl(`/users/${userId}/activate`), {
    method: 'PATCH',
  })

  if (res.ok) {
    await loadUsers()
  }
}

async function deleteUser(userId) {
  if (!confirm('Delete this user permanently?')) return

  const res = await fetchWithAuth(apiUrl(`/users/${userId}`), {
    method: 'DELETE',
  })

  if (res.ok) await loadUsers()
}

function buildFeedbackFilters() {
  const filters = {
    userId: feedbackFilters.value.userId || undefined,
    usedForRetrain:
      feedbackFilters.value.usedForRetrain === ''
        ? undefined
        : feedbackFilters.value.usedForRetrain,
  }

  if (feedbackFilters.value.dateFrom) {
    filters.dateFrom = localDateStartToUtcIso(feedbackFilters.value.dateFrom)
  }

  if (feedbackFilters.value.dateTo) {
    filters.dateTo = localDateEndToUtcIso(feedbackFilters.value.dateTo)
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

    retrainSuccess.value =
      `Retrain started for ${result.feedback_count} feedback entr${result.feedback_count === 1 ? 'y' : 'ies'}.`

    selectedFeedbackIds.value = new Set()
    await loadAdminFeedback()
  } catch (error) {
    retrainError.value = error.message ?? 'Failed to start retraining'
  } finally {
    retrainLoading.value = false
  }
}

function formatFeedbackDate(value) {
  return formatLocalDateTime(value)
}

onMounted(async () => {
  await loadUsers()
  await loadAdminFeedback()
})
</script>