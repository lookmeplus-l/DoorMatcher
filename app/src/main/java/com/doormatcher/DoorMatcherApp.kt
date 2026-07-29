package com.doormatcher

import android.app.Application
import android.content.Intent
import android.media.projection.MediaProjectionManager
import android.os.Build

class DoorMatcherApp : Application() {

    lateinit var mediaProjectionManager: MediaProjectionManager
        private set

    override fun onCreate() {
        super.onCreate()
        instance = this
        mediaProjectionManager = getSystemService(MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
    }

    companion object {
        lateinit var instance: DoorMatcherApp
            private set
    }
}
