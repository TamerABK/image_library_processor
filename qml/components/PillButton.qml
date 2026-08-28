import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Button {
    id: root

    property url iconSource: ""
    property string variant: "filled"

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: Math.max(96, pillLabel.implicitWidth + metrics.spacingMedium * 2)
    implicitHeight: metrics.toolbarButtonHeight
    leftPadding: metrics.spacingMedium
    rightPadding: metrics.spacingMedium
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

    contentItem: RowLayout {
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
            text: root.text
            color: root.foreground
            font.family: typography.buttonFamily
            font.pixelSize: typography.button
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

        Behavior on color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }
    }
}
