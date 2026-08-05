# What the app needs, and why

Android Studio's new-project wizard (Empty Activity + Compose) generates the AGP,
Kotlin, Compose BOM and core AndroidX entries for whatever toolchain version you
actually installed. **Let it.** Everything below is what you add on top.

Pin versions by letting Android Studio's "check for updates" resolve them rather than
copying numbers from here — these move monthly, and a stale pin is a confusing build
error rather than an obvious one.

## Dependencies to add

```toml
# gradle/libs.versions.toml  — merge into what the wizard generated

[versions]
media3       = "1.5.1"      # check for current
retrofit     = "2.11.0"
okhttp       = "4.12.0"
serialization = "1.7.3"
work         = "2.10.0"
room         = "2.6.1"
coil         = "2.7.0"
lifecycle    = "2.8.7"

[libraries]
# --- media: trimming on the way in, playback on the way out ---
media3-transformer = { module = "androidx.media3:media3-transformer", version.ref = "media3" }
media3-effect      = { module = "androidx.media3:media3-effect",      version.ref = "media3" }
media3-exoplayer   = { module = "androidx.media3:media3-exoplayer",   version.ref = "media3" }
media3-ui          = { module = "androidx.media3:media3-ui",          version.ref = "media3" }

# --- networking ---
retrofit           = { module = "com.squareup.retrofit2:retrofit", version.ref = "retrofit" }
okhttp             = { module = "com.squareup.okhttp3:okhttp",     version.ref = "okhttp" }
okhttp-logging     = { module = "com.squareup.okhttp3:logging-interceptor", version.ref = "okhttp" }
kotlinx-serialization-json = { module = "org.jetbrains.kotlinx:kotlinx-serialization-json", version.ref = "serialization" }
retrofit-kotlinx-serialization = { module = "com.jakewharton.retrofit:retrofit2-kotlinx-serialization-converter", version = "1.0.0" }

# --- background work that survives process death ---
work-runtime       = { module = "androidx.work:work-runtime-ktx", version.ref = "work" }

# --- the jobs list, which must outlive the app ---
room-runtime       = { module = "androidx.room:room-runtime", version.ref = "room" }
room-ktx           = { module = "androidx.room:room-ktx",     version.ref = "room" }
room-compiler      = { module = "androidx.room:room-compiler", version.ref = "room" }

# --- misc ---
coil-compose       = { module = "io.coil-kt:coil-compose", version.ref = "coil" }
lifecycle-viewmodel-compose = { module = "androidx.lifecycle:lifecycle-viewmodel-compose", version.ref = "lifecycle" }
```

## Why each one is here

| Library | Job | Why not something else |
|---|---|---|
| **media3-transformer** | Trim to 60 s, cap bitrate, strip location metadata before upload | Hardware-accelerated. Doing this yourself with MediaCodec is a week you don't need to spend. |
| **media3-exoplayer / ui** | Play the annotated result | `VideoView` chokes on codecs ExoPlayer handles. Same library family as the transformer. |
| **retrofit + okhttp** | Talk to the control plane | Retrofit for the JSON API; raw OkHttp for the video upload, because you want a streaming request body, not a serialized one. |
| **kotlinx-serialization** | JSON | Compile-time, reflection-free. Moshi/Gson also fine; this one is the Kotlin-native default. |
| **work-runtime** | The upload | **The load-bearing one.** Persists the job to disk and resumes after process death. Without it your upload dies when the user switches apps. |
| **room** | The jobs list | The annotated video may arrive while the app is dead. State has to be on disk, not in memory. DataStore would do for a v1 with one job at a time. |
| **coil-compose** | Video thumbnails | Handles the bitmap lifecycle so you don't leak. |

## Deliberately NOT included

- **No image-loading of the model, no TFLite, no ONNX, no OpenCV.** Inference is server-side. This is what that decision bought: the entire on-device ML stack disappears, and with it ~30 MB of APK and the ByteTrack-in-Kotlin rewrite.
- **No Hilt/Dagger for v1.** Manual construction is fine at this size, and DI is a lot of concepts to add on a first Android project.
- **No `READ_MEDIA_VIDEO` permission.** The system Photo Picker needs no permission at all, and requesting broad media access for an app that reads one user-chosen file is the single most likely Play rejection for this app.

## Manifest permissions

```xml
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_DATA_SYNC" />
```

That's the whole list. No camera, no storage, no media permissions — the Photo Picker
covers video selection and `MediaStore`/`ACTION_CREATE_DOCUMENT` covers saving the
result.

The foreground-service permission needs a **Play Console declaration including a video
demonstrating the feature**. Record it during closed testing; it surprises people.

## SDK levels

```kotlin
compileSdk = 36
targetSdk  = 36     // mandatory for new apps from 31 August 2026
minSdk     = 26     // Android 8.0 — Media3 and the Photo Picker backport are happy here
```

`targetSdk = 36` is not optional if you submit after 31 August. Build against it from
the first commit rather than bumping later — API 36 brings behaviour changes
(edge-to-edge enforcement, tighter background limits) that you want to hit during
testing, not after a rejection.
