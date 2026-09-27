import QtQuick
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "loadingPage"

    // Phase 3 presentation route only. The owner must explicitly change routes;
    // automatic completion awaits real startup orchestration, not a splash timer.

    Room36Theme.Theme { id: theme }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    Rectangle {
        anchors.fill: parent
        color: theme.mainBlue
    }

    Image {
        anchors.fill: parent
        source: theme.backgroundUrl("loading_mesh.svg")
        cache: true
        fillMode: Image.PreserveAspectCrop
    }

    Image {
        id: logo
        anchors.centerIn: parent
        width: Math.min(parent.width * 0.661, 1268.67)
        height: width / 9.059
        source: theme.brandingUrl("logo_loading_white.svg")
        cache: true
        fillMode: Image.PreserveAspectFit
        opacity: 0

        Component.onCompleted: opacity = 1

        Behavior on opacity {
            NumberAnimation {
                duration: 380
                easing.type: animations.standardEasing
            }
        }
    }
}
