package com.doormatcher.ui

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.util.DisplayMetrics
import android.view.Gravity
import android.view.WindowManager
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.NotificationCompat
import com.doormatcher.R
import com.doormatcher.data.DoorDatabase
import com.doormatcher.native.DoorMatcherNative
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlin.math.abs

/**
 * 悬浮窗服务
 * 负责：
 * 1. 创建悬浮窗（显示录制状态 + 最佳匹配结果）
 * 2. 持续截取游戏屏幕
 * 3. 运行比对算法
 * 4. 更新悬浮窗结果
 */
class FloatingService : Service() {

    private lateinit var windowManager: WindowManager
    private lateinit var floatingView: android.view.View
    private lateinit var db: DoorDatabase
    private lateinit var scope: CoroutineScope

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null
    private var isCapturing = false

    private val handler = Handler(Looper.getMainLooper())
    private var lastMatch = ""
    private var consecutiveCount = 0
    private var lastBitmap: Bitmap? = null

    companion object {
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_RESULT_DATA = "result_data"
        const val EXTRA_DOOR_DB_PATH = "door_db_path"
        const val CHANNEL_ID = "door_matcher_capture"
        const val NOTIFY_ID = 1

        // 连续多少帧匹配同一张门图才确认（防抖）
        private const val CONFIRM_THRESHOLD = 3
        // 比对间隔（毫秒）
        private const val MATCH_INTERVAL_MS = 500L
    }

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(WINDOW_SERVICE) as WindowManager
        db = DoorDatabase(this)
        scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

        createNotificationChannel()
        startForeground(NOTIFY_ID, createNotification("录制启动中..."))
        createFloatingWindow()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val resultCode = intent?.getIntExtra(EXTRA_RESULT_CODE, -1) ?: -1
        @Suppress("DEPRECATION")
        val resultData: Intent? = intent?.getParcelableExtra(EXTRA_RESULT_DATA)

        if (resultCode != -1 && resultData != null) {
            startProjection(resultCode, resultData)
        }
        return START_STICKY
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "屏幕录制",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "门图识别服务"
            }
            val nm = getSystemService(NotificationManager::class.java)
            nm.createNotificationChannel(channel)
        }
    }

    private fun createNotification(text: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("🚪 门图识别")
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    private fun updateNotification(text: String) {
        val nm = getSystemService(NotificationManager::class.java)
        nm.notify(NOTIFY_ID, createNotification(text))
    }

    private fun createFloatingWindow() {
        floatingView = android.view.LayoutInflater.from(this)
            .inflate(R.layout.floating_window, null)

        val params = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                @Suppress("DEPRECATION")
                WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                    WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.START
            x = 50
            y = 200
        }

        windowManager.addView(floatingView, params)

        floatingView.findViewById<android.view.View>(R.id.btn_close).setOnClickListener {
            stopSelf()
        }
    }

    private fun updateFloatingView(bestMatch: String, confidence: Float, icon: Bitmap? = null) {
        handler.post {
            floatingView.findViewById<TextView>(R.id.tv_match_result)?.text =
                "🎯 $bestMatch\n${"%.1f".format(confidence * 100)}%"
            icon?.let {
                floatingView.findViewById<ImageView>(R.id.iv_preview)?.setImageBitmap(it)
            }
        }
    }

    private fun startProjection(resultCode: Int, resultData: Intent) {
        val pm = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = pm.getMediaProjection(resultCode, resultData)

        val metrics = DisplayMetrics()
        @Suppress("DEPRECATION")
        windowManager.defaultDisplay.getMetrics(metrics)

        // 横屏游戏：宽>高，取短边作为高度
        val screenWidth = metrics.widthPixels
        val screenHeight = metrics.heightPixels

        // 只截取右半边（对应游戏界面右侧）
        val captureWidth = screenWidth / 2
        val captureHeight = screenHeight

        imageReader = ImageReader.newInstance(
            captureWidth, captureHeight, PixelFormat.RGBA_8888, 2
        )

        virtualDisplay = mediaProjection?.createVirtualDisplay(
            "DoorMatcherCapture",
            captureWidth, captureHeight, metrics.densityDpi,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            imageReader?.surface, null, null
        )

        isCapturing = true

        // 加载数据库
        scope.launch {
            db.load()
            startMatching()
        }
    }

    private fun startMatching() {
        val reader = imageReader ?: return

        handler.post {
            updateNotification("正在识别...")
        }

        // 比对循环
        handler.post(object : Runnable {
            override fun run() {
                if (!isCapturing) return

                val image = reader.acquireLatestImage()
                if (image != null) {
                    try {
                        processFrame(image.planes[0].buffer, image.width, image.height)
                    } catch (e: Exception) {
                        e.printStackTrace()
                    } finally {
                        image.close()
                    }
                }

                handler.postDelayed(this, MATCH_INTERVAL_MS)
            }
        })
    }

    private fun processFrame(buffer: java.nio.ByteBuffer, width: Int, height: Int) {
        // 转 Bitmap
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(buffer)

        // 若需要从右半边取中间区域
        // 实际游戏中，右半边可能还包含一些空白，需要更精细的裁剪
        val gameRegionX = width / 4
        val gameRegionW = width / 2
        val cropBitmap = Bitmap.createBitmap(
            bitmap, gameRegionX, 0, gameRegionW, height
        )
        bitmap.recycle()

        // 提取特征
        val regionPixels = DoorMatcherNative.extractRegion(
            cropBitmap, 0f, 1f, 0.25f, 0.75f, 450, 750
        )
        val regionHist = DoorMatcherNative.computeHistogram(regionPixels, 450, 750)

        // 构造门图特征
        val features = db.doors.map { door ->
            DoorMatcherNative.DoorFeatures(
                filename   = door.filename,
                fullWidth  = door.fullBitmap.width,
                fullHeight = door.fullBitmap.height,
                regionRGB  = door.regionRGB,
                histogram  = door.histogram,
                fullMean   = floatArrayOf(),
                fullVar    = floatArrayOf()
            )
        }

        // 比对
        val results = DoorMatcherNative.match(regionPixels, regionHist, features)
        val best = results.firstOrNull() ?: return

        // 防抖逻辑
        if (best.filename == lastMatch) {
            consecutiveCount++
        } else {
            consecutiveCount = 1
            lastMatch = best.filename
        }

        if (consecutiveCount >= CONFIRM_THRESHOLD) {
            val door = db.doors.find { it.filename == best.filename }
            updateFloatingView(best.filename, best.combined, door?.regionBitmap)
            updateNotification("🎯 ${best.filename} (${"%.1f".format(best.combined * 100)}%)")
        }

        cropBitmap.recycle()
    }

    override fun onDestroy() {
        super.onDestroy()
        isCapturing = false
        handler.removeCallbacksAndMessages(null)
        try {
            windowManager.removeView(floatingView)
        } catch (e: Exception) { }
        virtualDisplay?.release()
        mediaProjection?.stop()
        scope.cancel()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
