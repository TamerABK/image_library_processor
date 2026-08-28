import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Control {
    id: root

    property alias text: input.text
    property alias placeholderText: input.placeholderText
    property url iconSource: theme.iconUrl("search.svg")

    signal accepted(string text)
    signal searchAccepted(string text)

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.searchFieldWidth
    implicitHeight: metrics.toolbarButtonHeight
    leftPadding: metrics.spacingMedium
    rightPadding: metrics.spacingMedium
    topPadding: 0
    bottomPadding: 0
    focusPolicy: Qt.ClickFocus

    function forceInputFocus() {
        input.forceActiveFocus()
    }

    contentItem: RowLayout {
        spacing: metrics.spacingSmall

        Image {
            Layout.preferredWidth: metrics.iconSizeMedium
            Layout.preferredHeight: metrics.iconSizeMedium
            source: root.iconSource
            fillMode: Image.PreserveAspectFit
            opacity: root.enabled ? 1 : 0.36
        }

        TextField {
            id: input
            Layout.fillWidth: true
            Layout.fillHeight: true
            background: Item {}
            color: root.enabled ? theme.black : theme.disabledText
            placeholderTextColor: theme.mutedText
            font.family: typography.bodyFamily
            font.pixelSize: typography.body
            selectByMouse: true
            verticalAlignment: TextInput.AlignVCenter
            leftPadding: 0
            rightPadding: 0
            topPadding: 0
            bottomPadding: 0
            enabled: root.enabled

            onAccepted: {
                root.accepted(text)
                root.searchAccepted(text)
            }
        }
    }

    background: Rectangle {
        radius: height / 2
        color: root.enabled ? theme.panelRaisedFill : theme.disabledFill
        border.width: input.activeFocus ? 2 : 1
        border.color: input.activeFocus ? theme.focusRing : theme.separator

        Behavior on border.color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }
    }
}
