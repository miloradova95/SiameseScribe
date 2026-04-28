<template>
  <nav class="flex items-center justify-between px-12 py-8 bg-[#f7f1eb] text-[#2b211d]">
    
    <router-link
      to="/help"
      class="text-[23px] uppercase tracking-[0.28em] font-semibold transition hover:text-[#c53114]"
    >
      peuafleu
    </router-link>

    <div class="flex items-center gap-12 text-[23px]">
      <router-link to="/browse" class="transition hover:text-[#c53114]" active-class="text-[#c53114]">
        Browse
      </router-link>
      <router-link to="/upload" class="transition hover:text-[#c53114]" active-class="text-[#c53114]">
        Upload
      </router-link>
      <router-link to="/profile" class="transition hover:text-[#c53114]" active-class="text-[#c53114]">
        Profile
      </router-link>
      <router-link to="/help" class="transition hover:text-[#c53114]" active-class="text-[#c53114]">
        Help
      </router-link>
      <router-link v-if="authStore.isAdmin" to="/admin" class="hover:text-[#c53114] transition">
        Admin
      </router-link>
      <button
        v-if="authStore.isLoggedIn"
        @click="authStore.logout(); $router.push('/login')"
        class="hover:text-[#c53114] transition"
      >
        Logout
      </button>
    </div>

  </nav>
</template>

<script setup>
import { useAuthStore } from '../stores/auth'
const authStore = useAuthStore()
</script>