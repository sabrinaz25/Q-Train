from qtrain.android import parse_ungranted, parse_event, parse_resumed


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


APP = "com.ichi2.anki"
# (package, attributes, bounds) for each element under the root window
ELEMENTS = [
    (APP, 'clickable="true"', "[0,0][1080,1920]"),               # full-screen container
    (APP, 'clickable="true"', "[100,1500][980,1650]"),           # a button
    (APP, 'checkable="true"', "[900,400][1000,500]"),            # a checkbox
    (APP, 'clickable="true" enabled="false"', "[100,1700][980,1850]"),  # disabled button
    ("com.android.systemui", 'clickable="true"', "[0,0][120,80]"),     # status bar icon
    (APP, 'clickable="false"', "[100,200][980,300]"),            # plain text
]
DUMP = ("<?xml version='1.0' encoding='UTF-8'?><hierarchy rotation=\"0\">"
        f'<node package="{APP}" clickable="false" bounds="[0,0][1080,1920]">'
        + "".join(f'<node package="{pkg}" {attrs} bounds="{b}"/>' for pkg, attrs, b in ELEMENTS)
        + "</node></hierarchy>")


def test_parse_clickables_keeps_only_the_apps_usable_elements():
    from qtrain.android import parse_clickables
    boxes, display = parse_clickables(DUMP, "com.ichi2.anki")
    assert display == (1080, 1920)
    # the button and the checkbox; not the full-screen container, the disabled button,
    # the system UI icon or the plain text
    assert boxes == [(100, 1500, 980, 1650), (900, 400, 1000, 500)]
    assert parse_clickables("not xml", "x") == ([], None)


PACKAGE_DUMP = """
    requested permissions:
      android.permission.INTERNET
      android.permission.WRITE_EXTERNAL_STORAGE
      android.permission.CAMERA
      android.permission.POST_NOTIFICATIONS
    install permissions:
      android.permission.INTERNET: granted=true
    User 0: ceDataInode=1 installed=true
      runtime permissions:
        android.permission.CAMERA: granted=false, flags=[ USER_SENSITIVE ]
        android.permission.POST_NOTIFICATIONS: granted=true, flags=[ USER_SET ]
"""


def test_parse_ungranted_includes_requested_permissions_with_no_state_yet():
    # a freshly wiped app on an older Android lists WRITE_EXTERNAL_STORAGE as requested
    # but prints no state for it; it still needs granting
    assert parse_ungranted(PACKAGE_DUMP) == ["android.permission.CAMERA",
                                             "android.permission.WRITE_EXTERNAL_STORAGE"]
