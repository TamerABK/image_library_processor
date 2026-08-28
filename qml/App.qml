import QtQuick
import QtQuick.Controls
import "theme" as Room36Theme
import "dev" as Dev

ApplicationWindow {
    id: window

    property var appBridge: null

    visible: true
    width: 1440
    height: 900
    minimumWidth: metrics.minimumWindowWidth
    minimumHeight: metrics.minimumWindowHeight
    title: "Room 36 Component Gallery"
    color: theme.workspaceBackground

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }

    Dev.ComponentGallery {
        anchors.fill: parent
        bridge: window.appBridge
    }
}
