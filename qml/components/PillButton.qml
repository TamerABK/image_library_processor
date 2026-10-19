import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Button {
    id: root

    property url iconSource: ""
    property string variant: "filled"
    property bool compact: false
    property bool vignette: false

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: root.compact
        ? Math.max(72, compactLabel.implicitWidth + 16)
        : Math.max(96, pillLabel.implicitWidth + metrics.spacingMedium * 2)
    implicitHeight: metrics.toolbarButtonHeight
    leftPadding: root.compact ? 8 : metrics.spacingMedium
    rightPadding: root.compact ? 8 : metrics.spacingMedium
    topPadding: 0
    bottomPadding: 0
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    readonly property color normalFill: {
        if (variant === "accent") {
            return theme.mainBlue
        }
        if (variant === "translucent") {
            return theme.selectedFill
        }
        return theme.neonGreen
    }

    readonly property color foreground: {
        if (!enabled) {
            return theme.disabledText
        }
        return variant === "accent" ? theme.white : theme.black
    }

    contentItem: Item {
        RowLayout {
            anchors.fill: parent
            visible: !root.compact
            spacing: metrics.spacingTiny

            Image {
                Layout.preferredWidth: metrics.iconSizeSmall
                Layout.preferredHeight: metrics.iconSizeSmall
                source: root.iconSource
                visible: String(root.iconSource).length > 0
                fillMode: Image.PreserveAspectFit
                opacity: root.enabled ? 1 : 0.38
            }

            Text {
                id: pillLabel
                Layout.fillWidth: true
                Layout.fillHeight: true
                text: root.text
                color: root.foreground
                font.family: typography.buttonFamily
                font.pixelSize: typography.button
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
                elide: Text.ElideRight
            }
        }

        Text {
            id: compactLabel
            objectName: "pillCompactLabel"
            anchors.fill: parent
            visible: root.compact
            text: root.text
            color: root.foreground
            font.family: typography.buttonFamily
            font.pixelSize: typography.detail
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            elide: Text.ElideRight
        }
    }

    background: Rectangle {
        radius: height / 2
        color: !root.enabled
            ? theme.disabledFill
            : root.down
                ? theme.pressedFill
                : root.hovered || root.activeFocus
                    ? (root.variant === "filled" ? theme.lightBlue : theme.hoverFill)
                    : root.normalFill
        border.width: root.variant === "translucent" || root.activeFocus ? 1 : 0
        border.color: root.activeFocus ? theme.focusRing : theme.separator

        Rectangle {
            objectName: "pillVignetteOuter"
            anchors.fill: parent
            radius: parent.radius
            color: "transparent"
            border.width: 2
            border.color: Qt.rgba(0, 0, 0, 0.13)
            visible: root.vignette && root.enabled && !root.activeFocus
        }
        Rectangle {
            anchors.fill: parent
            anchors.margins: 2
            radius: Math.max(0, parent.radius - 2)
            color: "transparent"
            border.width: 1
            border.color: Qt.rgba(0, 0, 0, 0.055)
            visible: root.vignette && root.enabled && !root.activeFocus
        }

        Behavior on color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }
    }
}
