import { createRouter, createWebHistory } from 'vue-router'

import LoginPage from '../pages/LoginPage.vue'
import SignupPage from '../pages/SignupPage.vue'
import HomePage from '../pages/HomePage.vue'
import UploadPage from '../pages/UploadPage.vue'
import ProfilePage from '../pages/ProfilePage.vue'
import BrowsePage from '../pages/BrowsePage.vue'


const routes = [
  { path: '/', redirect: '/login' },
  { path: '/login', name: 'login', component: LoginPage },
  { path: '/signup', name: 'signup', component: SignupPage },
  { path: '/home', name: 'home', component: HomePage },
  { path: '/upload', name: 'upload', component: UploadPage },
  { path: '/profile', name: 'profile', component: ProfilePage },
  { path: '/browse', name: 'browse', component: BrowsePage },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router