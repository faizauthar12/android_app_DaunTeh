package com.android.gunadarma.daunteh.ui.splashscreen

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.ViewPropertyAnimator
import com.android.gunadarma.daunteh.R
import com.android.gunadarma.daunteh.databinding.ActivitySplashscreenBinding
import com.android.gunadarma.daunteh.ui.main.MainActivity

class SplashscreenActivity : AppCompatActivity() {
    private val time: Long = 1100
    private val alpha0 = 0f
    private val alpha1 = 1f
    private var propertyAnim: ViewPropertyAnimator? = null

    private lateinit var binding: ActivitySplashscreenBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySplashscreenBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.ivSplash.alpha = alpha0
        propertyAnim = binding.ivSplash.animate().setDuration(time).alpha(alpha1).withEndAction {
            Intent(this, MainActivity::class.java).apply {
                startActivity(this)
                finish()
            }
        }
    }

    override fun onDestroy() {
        propertyAnim?.cancel()
        super.onDestroy()
    }
}