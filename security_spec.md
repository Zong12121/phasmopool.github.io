# Security Specification (Phase 0 TDD)

## 1. Data Invariants
1. **User Profile Isolation**: `/users/{userId}/private/profile` contains PII (`email`, `displayName`, `photoURL`). Only the authenticated verified owner (`request.auth.uid == userId`) may read, create, or update it.
2. **Owner-Bound Mode States**: `/users/{userId}/modeStates/{modeId}` belongs strictly to `{userId}` where `data.uid == userId`, `modeId == data.mode`, and `modeId` is one of `['navigation', 'balance', 'parkour', 'biomechanics', 'manual_car', 'drone']`.
3. **Owner-Bound Checkpoints**: `/users/{userId}/checkpoints/{checkpointId}` belongs strictly to `{userId}` where `data.uid == userId`, `checkpointId` matches `^[a-zA-Z0-9_\-]+$`, and genome arrays are strictly bounded (`wIH.size() <= 1024`, `wHO.size() <= 512`, `bH.size() <= 64`, `bO.size() <= 32`).
4. **Temporal & Immutability Integrity**: `createdAt` and `uid` cannot be altered after creation; `createdAt == request.time` on create; `updatedAt == request.time` on update.

## 2. The "Dirty Dozen" Adversarial Payloads (Must Return PERMISSION_DENIED)
1. Unauthenticated read of `/users/alice/private/profile`.
2. Authenticated user `bob` reading `/users/alice/private/profile` (PII leak test).
3. Unverified email (`email_verified: false`) writing `/users/alice/modeStates/biomechanics`.
4. Identity spoofing: `bob` creating `/users/bob/modeStates/biomechanics` with `uid: "alice"`.
5. Path-key mismatch: `alice` creating `/users/alice/modeStates/parkour` with `mode: "biomechanics"`.
6. Shadow field injection: `alice` creating `/users/alice/modeStates/biomechanics` with extra `isAdmin: true`.
7. Oversized ID poisoning: `checkpointId` of length 300 characters or invalid characters `../admin`.
8. Oversized string/array DoS: `name` of 5000 chars or `history` array of 200 entries (`> 40`).
9. Client timestamp forgery: `updatedAt` set to a past timestamp instead of `request.time`.
10. Immutable field mutation: `alice` updating `uid` or `createdAt` on `/users/alice/checkpoints/cp1`.
11. Unbounded list query / cross-tenant list on `/users/alice/checkpoints` by `bob`.
12. Invalid mode enum `mode: "hacked_mode"` or negative `generation: -5`.
