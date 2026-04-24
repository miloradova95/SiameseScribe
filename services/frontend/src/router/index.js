import { createRouter, createWebHistory } from 'vue-router'

import LoginPage from '@/views/LoginPage.vue'
import SignupPage from '@/views/SignupPage.vue'
import HomePage from '@/views/HomePage.vue'
import UploadPage from '@/views/UploadPage.vue'
import ProfilePage from '@/views/ProfilePage.vue'
import BrowsePage from '@/views/BrowsePage.vue'
import AnnotatePage from '@/views/AnnotatePage.vue'


const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', name: 'login', component: LoginPage },
  { path: '/signup', name: 'signup', component: SignupPage },
  { path: '/home', name: 'home', component: HomePage },
  { path: '/upload', name: 'upload', component: UploadPage },
  { path: '/profile', name: 'profile', component: ProfilePage },
  { path: '/browse', name: 'browse', component: BrowsePage,},
  { path: '/browse/:fileName', name: 'annotate', component: AnnotatePage, props: true,}
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router