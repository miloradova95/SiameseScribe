<template>
  <div class="min-h-screen flex items-center justify-center bg-[#f7f1eb]">
    <div class="w-full max-w-sm p-8 bg-white rounded shadow">
      <h1 class="text-2xl font-semibold text-[#2b211d] mb-6">Sign in</h1>
      <form @submit.prevent="submit">
        <div class="mb-4">
          <label class="block text-sm mb-1">Username</label>
          <input
            v-model="username"
            type="text"
            class="w-full border border-gray-300 rounded px-3 py-2 text-sm"
            required
            autofocus
          />
        </div>
        <div class="mb-6">
          <label class="block text-sm mb-1">Password</label>
          <input
            v-model="password"
            type="password"
            class="w-full border border-gray-300 rounded px-3 py-2 text-sm"
            required
          />
        </div>
        <p v-if="error" class="text-red-600 text-sm mb-4">{{ error }}</p>
        <button
          type="submit"
          :disabled="loading"
          class="w-full bg-[#2b211d] text-white py-2 rounded text-sm hover:bg-[#c53114] transition"
        >
          {{ loading ? 'Signing in…' : 'Sign in' }}
        </button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const authStore = useAuthStore()
const router = useRouter()

const username = ref('')
const password = ref('')
const error = ref('')
const loading = ref(false)

async function submit() {
  error.value = ''
  loading.value = true
  try {
    await authStore.login(username.value, password.value)
    router.push('/home')
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>