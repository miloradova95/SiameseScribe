export function formatLocalDateTime(value) {
  return new Date(value).toLocaleString()
}

export function localDateStartToUtcIso(dateValue) {
  if (!dateValue) return undefined

  const start = new Date(`${dateValue}T00:00:00`)
  return start.toISOString()
}

export function localDateEndToUtcIso(dateValue) {
  if (!dateValue) return undefined

  const end = new Date(`${dateValue}T23:59:59.999`)
  return end.toISOString()
}
