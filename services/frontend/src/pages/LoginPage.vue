<template>
  <div class="min-h-screen bg-[#f7f1eb] flex items-center text-[#5b4033]">
    
    <div class="mx-auto w-full max-w-[1400px] px-16 grid grid-cols-[1fr_420px] items-center gap-20">
      
      <section>
        <h1
          class="font-display text-[96px] font-medium uppercase leading-none tracking-[0.08em]"
        >
          PEU A FLEU
        </h1>

        <p class="mt-6 max-w-[620px] text-[24px] leading-snug">
          Lorem ipsum dolor sit amet, consectetur adipiscing elit.
          Phasellus pretium ex vitae ipsum egestas,
        </p>
      </section>


      <section
        class="rounded-[30px] bg-[#5b4033] px-12 py-14 text-[#fbf8f5] shadow-xl"
      >
        <form @submit.prevent="submit">

          <div class="mb-5">
            <label class="mb-2 block text-[18px]">Username</label>
            <input
              v-model="username"
              type="text"
              required
              autofocus
              class="h-14 w-full rounded-[22px] border border-[#fbf8f5] bg-transparent px-5 text-[17px] text-[#fbf8f5] outline-none focus:ring-2 focus:ring-white/40"
            />
          </div>

          <div class="mb-5">
            <label class="mb-2 block text-[18px]">Password</label>
            <input
              v-model="password"
              type="password"
              required
              class="h-14 w-full rounded-[22px] border border-[#fbf8f5] bg-transparent px-5 text-[17px] text-[#fbf8f5] outline-none focus:ring-2 focus:ring-white/40"
            />
          </div>

          <label class="mb-6 flex items-center gap-3 text-[17px]">
            <input type="checkbox" class="h-5 w-5 rounded border-white" />
            Remember me
          </label>

          <p v-if="error" class="mb-4 text-sm text-red-200">
            {{ error }}
          </p>

          <button
            type="submit"
            :disabled="loading"
            class="w-full rounded-full bg-[#e9e1d9] py-4 font-display text-[20px] uppercase tracking-[0.35em] text-[#5b4033] shadow-md transition hover:bg-[#dcd2c8] disabled:opacity-60"
          >
            {{ loading ? 'Signing in...' : 'Login' }}
          </button>

          <p class="mt-5 text-[15px] leading-snug">
            By logging in, you agree to the<br />
            <a href="#" class="text-[#4db7ff] hover:underline">Terms of Use</a>
            and
            <a href="#" class="text-[#4db7ff] hover:underline">Privacy Policy</a>
          </p>

        </form>
      </section>

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
    router.push('/help')
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}
</script>