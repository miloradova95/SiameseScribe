<template>
  <nav class="flex items-center justify-between px-12 py-8 bg-[#f7f1eb] text-[#2b211d]">
    
    <router-link
      to="/home"
      class="text-[23px] uppercase tracking-[0.28em] font-semibold"
    >
      peuafleu
    </router-link>

    <div class="flex items-center gap-12 text-[23px]">
      <router-link to="/browse" class="transition hover:text-[#c53114]">
        Browse
      </router-link>
      <router-link to="/upload" class="transition hover:text-[#c53114]">
        Upload
      </router-link>
      <router-link to="/profile" class="transition hover:text-[#c53114]">
        Profile
      </router-link>
      <router-link to="/home" class="transition hover:text-[#c53114]">
        Home
      </router-link>
    </div>

  </nav>
</template>