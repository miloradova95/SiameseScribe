<template>
  <div class="min-h-screen bg-[#fbfaf8] px-6 py-8 text-[#5b4034]">
    <div class="mx-auto max-w-6xl">
      <div class="mb-8 flex items-start justify-between gap-4">
        <h1 class="text-6xl font-bold tracking-tight text-[#5b4034]">
          Admin
        </h1>

        <button type="button"
          class="inline-flex h-9 items-center justify-center gap-2.5 whitespace-nowrap rounded-full bg-[#5b4034] px-5 text-sm font-medium text-white transition hover:bg-[#c53114]"
          @click="showCreateUser = true">
          <PlusIcon />
          <span class="block leading-none">Create Member</span>
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
        <button type="button" class="pb-3 text-sm font-semibold" :class="activeTab === 'members'
          ? 'border-b-2 border-[#5b4034] text-[#5b4034]'
          : 'text-[#9a867c]'" @click="activeTab = 'members'">
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
          Model Finetuning
          <span class="ml-1 rounded-full bg-[#ead8b9] px-2 py-0.5 text-xs text-[#a7441d]">
            {{ feedbackItems.length }}
          </span>
        </button>

        <button
          type="button"
          class="pb-3 text-sm font-semibold"
          :class="activeTab === 'ml'
            ? 'border-b-2 border-[#5b4034] text-[#5b4034]'
            : 'text-[#9a867c]'"
          @click="activeTab = 'ml'"
        >
          ML Runs
        </button>
      </div>

      <section v-if="activeTab === 'members'">
        <div
          class="grid grid-cols-[1.5fr_0.8fr_0.8fr_1.2fr] border-b border-[#e5d8d1] px-3 py-4 text-xs font-semibold text-[#9a867c]">
          <div>User</div>
          <div>Status</div>
          <div>Role</div>
          <div></div>
        </div>

        <div v-for="u in users" :key="u.id"
          class="grid grid-cols-[1.5fr_0.8fr_0.8fr_1.2fr] items-center border-b px-3 py-5 transition" :class="u.id === authStore.user?.id
            ? 'bg-[#fff7ed] border-[#e3b5a8]'
            : 'border-[#e3b5a8]'">

          <div class="flex items-center gap-4">
            <div
              class="flex h-10 w-10 items-center justify-center rounded-full bg-[#e7f2ff] text-sm font-bold text-[#1976d2]">
              {{ u.username?.slice(0, 2).toUpperCase() }}
            </div>

            <div>

              <div class="font-semibold text-[#5b4034]">{{ u.username }}</div>
              <div class="text-sm text-[#9a867c]">{{ u.email }}</div>
            </div>
          </div>

          <div>
            <span class="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold" :class="u.is_active
              ? 'bg-[#dff5e3] text-[#2e7d32] ring-1 ring-[#2e7d32]/20'
              : 'bg-[#f1e8e2] text-[#9a867c]'">
              <span class="h-2 w-2 rounded-full" :class="u.is_active ? 'bg-[#2e7d32]' : 'bg-[#9a867c]'"></span>

              {{ u.is_active ? 'Active' : 'Deactivated' }}
            </span>
          </div>

          <div class="text-sm text-[#5b4034]">{{ u.role }}</div>

          <div class="flex gap-4">
            <button v-if="u.is_active && u.id !== authStore.user?.id" @click="deactivate(u.id)"
              class="rounded-full border border-[#bba79d] px-4 py-1.5 text-sm text-[#5b4034] hover:bg-[#f1e8e2]">
              Deactivate
            </button>

            <button v-else-if="!u.is_active" @click="activate(u.id)"
              class="rounded-full border border-[#91b875] px-4 py-1.5 text-sm text-[#3f7b22] hover:bg-[#e3f1d8]">
              Activate
            </button>

            <span v-else class="rounded-full border border-[#91b875] px-4 py-1.5 text-sm text-[#3f7b22]">
              Current User
            </span>

            <button v-if="u.id !== authStore.user?.id" @click="deleteUser(u.id)"
              class="rounded-full border border-[#e3b5a8] px-4 py-1.5 text-sm text-[#c53114] hover:bg-[#f8e4df]">
              Delete
            </button>

          </div>
        </div>

        <p v-if="loadError" class="mt-4 text-sm text-red-600">{{ loadError }}</p>
        <p v-if="createError" class="mt-4 text-sm text-red-600">{{ createError }}</p>
        <p v-if="createSuccess" class="mt-4 text-sm text-green-600">{{ createSuccess }}</p>
      </section>

      <section v-if="activeTab === 'feedback'" class="space-y-6">

        <!-- Info banner -->
        <div class="rounded-2xl bg-[#ead8b9]/60 px-5 py-4 text-sm text-[#5b4034] space-y-1">
          <p class="font-semibold">Finetuning runs automatically.</p>
          <p class="text-[#8a756b]">
            The scheduler checks for new unused feedback every 15 minutes (every 7 days for development) and starts a run
            when at least one anchor patch has a "not similar" label. Use
            <span class="font-medium">Trigger Now</span> to run an immediate check without waiting.
            All runs are logged in the table below.
          </p>
        </div>

        <!-- Manual trigger -->
        <div class="flex flex-wrap items-center gap-3">
          <button
            type="button"
            class="rounded-full bg-[#c53114] px-5 py-2 text-sm text-white disabled:opacity-40 hover:bg-[#a02a10]"
            :disabled="triggerLoading || reembedStatus.in_progress"
            @click="triggerNow"
          >
            {{ triggerLoading ? 'Checking…' : reembedStatus.in_progress ? (reembedStatus.phase === 'evaluating' ? 'Evaluating…' : 'Embedding…') : 'Trigger Now' }}
          </button>
          <p v-if="triggerResult" class="text-sm" :class="triggerResult.status === 'triggered' ? 'text-green-700' : 'text-[#8a756b]'">
            {{ triggerResult.status === 'triggered'
              ? `Run #${triggerResult.run_id} queued.`
              : `Skipped — ${triggerResult.reason}` }}
          </p>
          <p v-if="triggerError" class="text-sm text-red-600">{{ triggerError }}</p>
        </div>

        <!-- Finetune run log -->
        <div>
          <div class="mb-2 flex items-center justify-between">
            <h2 class="text-sm font-semibold text-[#5b4034]">Recent Finetune Runs</h2>
            <button
              type="button"
              class="text-xs text-[#9a867c] underline hover:text-[#5b4034]"
              @click="loadFinetuneRuns"
            >Refresh</button>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-[#ded2ca] text-left text-xs text-[#9a867c]">
                  <th class="px-3 py-3">#</th>
                  <th class="px-3 py-3">Triggered</th>
                  <th class="px-3 py-3">Source</th>
                  <th class="px-3 py-3">Status</th>
                  <th class="px-3 py-3">Samples</th>
                  <th class="px-3 py-3">Triplets</th>
                  <th class="px-3 py-3">MLflow run</th>
                  <th class="px-3 py-3">Error</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="run in finetuneRuns"
                  :key="run.id"
                  class="border-b border-[#eee5df] align-top"
                >
                  <td class="px-3 py-3 text-[#9a867c]">{{ run.id }}</td>
                  <td class="whitespace-nowrap px-3 py-3">{{ formatFeedbackDate(run.triggered_at) }}</td>
                  <td class="px-3 py-3 capitalize">{{ run.trigger_source }}</td>
                  <td class="px-3 py-3">
                    <span
                      class="rounded-full px-2 py-0.5 text-xs font-medium"
                      :class="{
                        'bg-yellow-100 text-yellow-800': run.status === 'pending' || run.status === 'running',
                        'bg-green-100 text-green-800': run.status === 'completed',
                        'bg-red-100 text-red-800': run.status === 'failed',
                      }"
                    >{{ run.status }}</span>
                  </td>
                  <td class="px-3 py-3 text-xs text-[#8a756b]">
                    {{ run.t_real }}R / {{ run.t_aug }}A / {{ run.p_pos }}P
                    <span class="block text-[10px]">real / aug / pairs</span>
                  </td>
                  <td class="px-3 py-3">{{ run.triplets_used }}</td>
                  <td class="px-3 py-3 font-mono text-xs text-[#9a867c]">{{ run.mlflow_run_id?.slice(0, 8) ?? '—' }}</td>
                  <td class="break-all px-3 py-3 text-xs text-red-600">{{ run.error_msg ?? '' }}</td>
                </tr>
                <tr v-if="!finetuneRuns.length">
                  <td colspan="8" class="py-8 text-center text-[#9a867c]">No finetune runs yet.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Feedback log (read-only, for reference) -->
        <div>
          <div class="mb-2 text-sm font-semibold text-[#5b4034]">Feedback Log</div>
          <form
            @submit.prevent="loadAdminFeedback"
            class="mb-4 grid gap-4 rounded-2xl bg-white/70 p-5 md:grid-cols-4"
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
              <option value="false">Not yet used</option>
              <option value="true">Used in a run</option>
            </select>

            <div class="md:col-span-4 flex gap-3">
              <button
                type="submit"
                class="rounded-full bg-[#5b4034] px-5 py-2 text-sm text-white hover:bg-[#c53114]"
              >Apply Filters</button>
              <button
                type="button"
                class="rounded-full border border-[#bba79d] px-5 py-2 text-sm"
                @click="resetFeedbackFilters"
              >Reset</button>
            </div>
          </form>

          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-[#ded2ca] text-left text-xs text-[#9a867c]">
                  <th class="px-3 py-4">Date</th>
                  <th class="px-3 py-4">User</th>
                  <th class="px-3 py-4">Label</th>
                  <th class="px-3 py-4">Query Patch</th>
                  <th class="px-3 py-4">Result Patch</th>
                  <th class="px-3 py-4">Used in run</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="item in feedbackItems"
                  :key="item.id"
                  class="border-b border-[#eee5df] align-top"
                >
                  <td class="whitespace-nowrap px-3 py-4">{{ formatFeedbackDate(item.created_at) }}</td>
                  <td class="px-3 py-4">
                    <div class="font-semibold text-[#5b4034]">{{ item.username }}</div>
                    <div class="text-xs text-[#9a867c]">#{{ item.user_id }} • {{ item.user_email }}</div>
                  </td>
                  <td class="px-3 py-4">{{ item.label }}</td>
                  <td class="break-all px-3 py-4">{{ item.query_patch_file_name }}</td>
                  <td class="break-all px-3 py-4">{{ item.result_patch_file_name }}</td>
                  <td class="px-3 py-4">
                    <span
                      class="rounded-full px-2 py-0.5 text-xs"
                      :class="item.used_for_retrain ? 'bg-green-100 text-green-800' : 'bg-[#ead8b9] text-[#8a756b]'"
                    >{{ item.used_for_retrain ? 'yes' : 'pending' }}</span>
                  </td>
                </tr>
                <tr v-if="!feedbackLoading && !feedbackItems.length">
                  <td colspan="6" class="py-8 text-center text-[#9a867c]">
                    No feedback matched the current filters.
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
          <p v-if="feedbackError" class="mt-2 text-sm text-red-600">{{ feedbackError }}</p>
        </div>

      </section>

      <!-- ML Runs tab -->
      <section v-if="activeTab === 'ml'" class="space-y-6">

        <!-- Re-embedding status card -->
        <div class="rounded-2xl bg-white/70 p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h2 class="text-sm font-semibold text-[#5b4034]">Re-embedding Status</h2>
            <span
              class="rounded-full px-2 py-0.5 text-xs font-medium"
              :class="{
                'bg-yellow-100 text-yellow-800': reembedStatus.phase === 'embedding',
                'bg-blue-100 text-blue-800':    reembedStatus.phase === 'evaluating',
                'bg-green-100 text-green-800':  reembedStatus.phase === 'idle',
              }"
            >{{
              reembedStatus.phase === 'embedding' ? 'Embedding…' :
              reembedStatus.phase === 'evaluating' ? 'Evaluating…' : 'Idle'
            }}</span>
          </div>
          <div v-if="reembedStatus.in_progress" class="text-sm text-[#8a756b]">
            Started: {{ reembedStatus.started_at ? formatFeedbackDate(reembedStatus.started_at) : '—' }}
          </div>
          <div v-if="!reembedStatus.in_progress && reembedStatus.completed_at" class="text-sm text-[#8a756b]">
            Last completed: {{ formatFeedbackDate(reembedStatus.completed_at) }}
            <span v-if="reembedStatus.eval_precision_at_k != null" class="ml-3 font-medium text-[#5b4034]">
              P@5 {{ (reembedStatus.eval_precision_at_k * 100).toFixed(1) }}%
              &nbsp;·&nbsp; mAP {{ (reembedStatus.eval_mAP * 100).toFixed(1) }}%
            </span>
          </div>
          <a
            :href="mlflowUrl"
            target="_blank"
            rel="noopener"
            class="inline-block rounded-full border border-[#bba79d] px-4 py-1.5 text-xs text-[#5b4034] hover:bg-[#ead8b9]"
          >Open MLflow UI ↗</a>
        </div>

        <!-- Finetune run history -->
        <div>
          <div class="mb-2 flex items-center justify-between">
            <h2 class="text-sm font-semibold text-[#5b4034]">Finetune Run History</h2>
            <button
              type="button"
              class="text-xs text-[#9a867c] underline hover:text-[#5b4034]"
              @click="loadFinetuneRuns"
            >Refresh</button>
          </div>
          <div class="overflow-x-auto">
            <table class="w-full text-sm">
              <thead>
                <tr class="border-b border-[#ded2ca] text-left text-xs text-[#9a867c]">
                  <th class="px-3 py-3">#</th>
                  <th class="px-3 py-3">Triggered</th>
                  <th class="px-3 py-3">Source</th>
                  <th class="px-3 py-3">Status</th>
                  <th class="px-3 py-3">Samples</th>
                  <th class="px-3 py-3">Triplets</th>
                  <th class="px-3 py-3">P@5</th>
                  <th class="px-3 py-3">mAP</th>
                  <th class="px-3 py-3">MLflow</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="run in finetuneRuns" :key="run.id" class="border-b border-[#eee5df] align-top">
                  <td class="px-3 py-3 text-[#9a867c]">{{ run.id }}</td>
                  <td class="whitespace-nowrap px-3 py-3">{{ formatFeedbackDate(run.triggered_at) }}</td>
                  <td class="px-3 py-3 capitalize">{{ run.trigger_source }}</td>
                  <td class="px-3 py-3">
                    <span
                      class="rounded-full px-2 py-0.5 text-xs font-medium"
                      :class="{
                        'bg-yellow-100 text-yellow-800': ['pending','running','reembedding','evaluating'].includes(run.status),
                        'bg-green-100 text-green-800':  run.status === 'completed',
                        'bg-red-100 text-red-800':      run.status === 'failed',
                      }"
                    >{{ run.status }}</span>
                  </td>
                  <td class="px-3 py-3 text-xs text-[#8a756b]">
                    {{ run.t_real }}R / {{ run.t_aug }}A / {{ run.p_pos }}P
                  </td>
                  <td class="px-3 py-3">{{ run.triplets_used }}</td>
                  <td class="px-3 py-3">
                    {{ run.eval_precision_at_k != null ? (run.eval_precision_at_k * 100).toFixed(1) + '%' : '—' }}
                  </td>
                  <td class="px-3 py-3">
                    {{ run.eval_mAP != null ? (run.eval_mAP * 100).toFixed(1) + '%' : '—' }}
                  </td>
                  <td class="px-3 py-3 font-mono text-xs text-[#9a867c]">
                    {{ run.mlflow_run_id?.slice(0, 8) ?? '—' }}
                  </td>
                </tr>
                <tr v-if="!finetuneRuns.length">
                  <td colspan="9" class="py-8 text-center text-[#9a867c]">No finetune runs yet.</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

      </section>

    </div>

    <CreateUserModal :open="showCreateUser" @close="showCreateUser = false" @create="handleCreateUser" />
  </div>
</template>

<script setup>
import { computed, onMounted, onUnmounted, ref } from 'vue'
import { useAuthStore } from '../stores/auth'
import { fetchWithAuth, apiUrl } from '../lib/api'
import { formatLocalDateTime, localDateEndToUtcIso, localDateStartToUtcIso } from '../lib/date'
import { fetchAdminFeedback, triggerFinetuneRun, fetchFinetuneRuns, fetchReembedStatus } from '../services/patch-service'
import PlusIcon from '../components/PlusIcon.vue'
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

const finetuneRuns = ref([])
const triggerLoading = ref(false)
const triggerResult = ref(null)
const triggerError = ref('')

const reembedStatus = ref({ in_progress: false, phase: 'idle', started_at: null, completed_at: null, eval_precision_at_k: null, eval_mAP: null })
const mlflowUrl = import.meta.env.VITE_MLFLOW_URL ?? 'http://localhost:5000'
let reembedPollTimer = null

const feedbackFilters = ref({
  userId: '',
  dateFrom: '',
  dateTo: '',
  usedForRetrain: 'false',
})

const activeMembersCount = computed(() => users.value.filter((u) => u.is_active).length)
const deactivatedMembersCount = computed(() => users.value.filter((u) => !u.is_active).length)

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
  try {
    feedbackItems.value = await fetchAdminFeedback(buildFeedbackFilters())
  } catch (error) {
    feedbackError.value = error.message ?? 'Failed to load feedback'
  } finally {
    feedbackLoading.value = false
  }
}

async function loadFinetuneRuns() {
  try {
    finetuneRuns.value = await fetchFinetuneRuns()
  } catch {
    // non-blocking — table stays empty
  }
}

async function pollReembedStatus() {
  try {
    reembedStatus.value = await fetchReembedStatus()
  } catch {
    // non-blocking
  }
}

function startReembedPolling() {
  pollReembedStatus()
  reembedPollTimer = setInterval(async () => {
    await pollReembedStatus()
    // Also refresh run list so status badges update
    if (reembedStatus.value.in_progress) {
      await loadFinetuneRuns()
    }
  }, 10000)
}

function stopReembedPolling() {
  if (reembedPollTimer) {
    clearInterval(reembedPollTimer)
    reembedPollTimer = null
  }
}

function resetFeedbackFilters() {
  feedbackFilters.value = { userId: '', dateFrom: '', dateTo: '', usedForRetrain: 'false' }
  loadAdminFeedback()
}

async function triggerNow() {
  const confirmed = window.confirm(
    'Start a finetune run now?\n\n' +
    'After training completes, all patches will be re-embedded with the new model weights. ' +
    'This can take a significant some time (potentially hours) depending on dataset size. ' +
    'Search results will remain available throughout.'
  )
  if (!confirmed) return

  triggerLoading.value = true
  triggerResult.value = null
  triggerError.value = ''
  try {
    triggerResult.value = await triggerFinetuneRun()
    await loadFinetuneRuns()
  } catch (error) {
    triggerError.value = error.message ?? 'Failed to trigger finetune run'
  } finally {
    triggerLoading.value = false
  }
}

function formatFeedbackDate(value) {
  return formatLocalDateTime(value)
}

onMounted(async () => {
  await loadUsers()
  await Promise.all([loadAdminFeedback(), loadFinetuneRuns()])
  startReembedPolling()
})

onUnmounted(() => {
  stopReembedPolling()
})
</script>
