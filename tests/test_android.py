from qtrain.android import parse_event, parse_resumed


def test_resumed_activity_from_activity_manager():
    out = "    topResumedActivity=ActivityRecord{9f1c2 u0 com.android.settings/.Settings t41}\n"
    assert parse_resumed(out) == ("com.android.settings", "com.android.settings.Settings")
    old = "  mResumedActivity: ActivityRecord{1a2b u0 org.app/org.app.ui.MainActivity t3}"
    assert parse_resumed(old) == ("org.app", "org.app.ui.MainActivity")


def test_resumed_activity_from_window_manager():
    out = "  mCurrentFocus=Window{4e1 u0 com.example/com.example.Detail$Inner}"
    assert parse_resumed(out) == ("com.example", "com.example.Detail$Inner")
    assert parse_resumed("mCurrentFocus=null") is None


def test_crash_and_anr_events_for_the_package_only():
    crash = ("I/am_crash(  612): [3011,0,com.example,952745542,java.lang.NullPointerException,"
             "Attempt to invoke virtual method, with, commas,MainActivity.java,42,0]")
    assert parse_event(crash, "com.example") == ("crash", "java.lang.NullPointerException")
    assert parse_event(crash, "com.other") is None
    remote = "I/am_crash(  612): [3011,0,com.example:sync,0,java.lang.IllegalStateException,boom,X.java,1]"
    assert parse_event(remote, "com.example") == ("crash", "java.lang.IllegalStateException")
    anr = "I/am_anr  (  612): [0,3011,com.example,952745542,Input dispatching timed out]"
    assert parse_event(anr, "com.example") == ("anr", "Input dispatching timed out")
    assert parse_event("I/am_proc_start( 612): [0,3011,10123,com.example,activity,{}]", "com.example") is None
