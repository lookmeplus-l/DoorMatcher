package com.doormatcher.ui

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.media.projection.MediaProjectionManager
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.WindowInsetsControllerCompat
import androidx.lifecycle.lifecycleScope
import com.doormatcher.DoorMatcherApp
import com.doormatcher.R
import com.doormatcher.data.DoorDatabase
import com.doormatcher.databinding.ActivityMainBinding
import com.doormatcher.native.DoorMatcherNative
import com.doormatcher.tflite.DoorMatcherTFLite
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var db: DoorDatabase
    private lateinit var tflite: DoorMatcherTFLite

    private val REQUEST_MEDIA_PROJECTION = 1001

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        hideSystemUI()
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        db = DoorDatabase(this)
        tflite = DoorMatcherTFLite(this)

        setupUI()
        loadDatabase()
    }

    private fun hideSystemUI() {
        WindowCompat.setDecorFitsSystemWindows(window, false)
        WindowInsetsControllerCompat(window, window.decorView).let {
            it.hide(WindowInsetsCompat.Type.systemBars())
            it.systemBarsBehavior =
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        }
    }

    private fun setupUI() {
        binding.btnStartCapture.setOnClickListener {
            startScreenCapture()
        }

        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        binding.btnManualSelect.setOnClickListener {
            // TODO: 实现手动选图
            Toast.makeText(this, "手动选图功能开发中", Toast.LENGTH_SHORT).show()
        }
    }

    private fun loadDatabase() {
        binding.tvStatus.text = "正在加载门图数据库..."
        lifecycleScope.launch {
            db.load()
            withContext(Dispatchers.Main) {
                binding.tvStatus.text = "✅ 数据库已就绪 (${db.doors.size} 张门图)"
            }

            // 预提取特征向量
            db.doors.forEach { door ->
                val features = DoorMatcherNative.DoorFeatures(
                    filename   = door.filename,
                    fullWidth  = door.fullBitmap.width,
                    fullHeight = door.fullBitmap.height,
                    regionRGB  = door.regionRGB,
                    histogram  = door.histogram,
                    fullMean   = floatArrayOf(),
                    fullVar    = floatArrayOf()
                )
            }
        }
    }

    private fun startScreenCapture() {
        val app = application as DoorMatcherApp
        val pm: MediaProjectionManager = app.mediaProjectionManager
        startActivityForResult(pm.createScreenCaptureIntent(), REQUEST_MEDIA_PROJECTION)
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_MEDIA_PROJECTION && resultCode == Activity.RESULT_OK && data != null) {
            // 启动悬浮窗服务
            val serviceIntent = Intent(this, FloatingService::class.java).apply {
                putExtra(FloatingService.EXTRA_RESULT_CODE, resultCode)
                putExtra(FloatingService.EXTRA_RESULT_DATA, data)
                putExtra(FloatingService.EXTRA_DOOR_DB_PATH, db.doors.firstOrNull()?.filename ?: "")
            }
            startForegroundService(serviceIntent)

            // 同时启动识别
            binding.tvStatus.text = "🎮 录制已启动，请进入游戏"
            binding.btnStartCapture.text = "录制中..."
            binding.btnStartCapture.isEnabled = false
        }
    }
}
