import QtQuick
import QtQuick.Effects

Item {
    id: root

    property Item sourceItem: null
    property bool active: false
    property color glowColor: Qt.rgba(1, 1, 1, 0.50)
    property real glowBlur: 0.32
    property real glowScale: 1.02

    anchors.fill: sourceItem
    visible: active && sourceItem !== null
    enabled: visible

    MultiEffect {
        anchors.fill: parent
        source: root.sourceItem
        autoPaddingEnabled: true
        shadowEnabled: root.visible
        shadowColor: root.glowColor
        shadowOpacity: 0.50
        shadowBlur: root.glowBlur
        shadowScale: root.glowScale
        shadowHorizontalOffset: 0
        shadowVerticalOffset: 0
        blurEnabled: false
    }
}
