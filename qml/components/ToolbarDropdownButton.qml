pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Button {
    id: root

    property string label: text
    property url iconSource: ""
    property bool open: popup.opened
    property var options: ["Newest", "Oldest", "Largest"]

    signal optionSelected(string option)

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.toolbarDropdownWidth
    implicitHeight: metrics.toolbarButtonHeight
    leftPadding: metrics.spacingMedium
    rightPadding: metrics.spacingSmall
    hoverEnabled: true
    focusPolicy: Qt.StrongFocus

    onClicked: popup.opened ? popup.close() : popup.open()

    contentItem: RowLayout {
        spacing: metrics.spacingTiny

        Image {
            Layout.preferredWidth: metrics.iconSizeMedium
            Layout.preferredHeight: metrics.iconSizeMedium
            source: root.iconSource
            visible: String(root.iconSource).length > 0
            fillMode: Image.PreserveAspectFit
        }

        Text {
            Layout.fillWidth: true
            text: root.label
            color: root.enabled ? theme.black : theme.disabledText
            font.family: typography.buttonFamily
            font.pixelSize: typography.button
            elide: Text.ElideRight
            verticalAlignment: Text.AlignVCenter
        }

        Image {
            Layout.preferredWidth: metrics.iconSizeSmall
            Layout.preferredHeight: metrics.iconSizeSmall
            source: theme.iconUrl("chevron_down.svg")
            fillMode: Image.PreserveAspectFit
            rotation: popup.opened ? 180 : 0

            Behavior on rotation {
                NumberAnimation {
                    duration: animations.popup
                    easing.type: animations.standardEasing
                }
            }
        }
    }

    background: Rectangle {
        radius: metrics.toolbarButtonRadius
        color: !root.enabled
            ? theme.disabledFill
            : root.down || popup.opened
                ? theme.pressedFill
                : root.hovered || root.activeFocus
                    ? theme.hoverFill
                    : theme.panelRaisedFill
        border.width: popup.opened || root.activeFocus ? 2 : 1
        border.color: popup.opened || root.activeFocus ? theme.focusRing : theme.separator

        Behavior on color {
            ColorAnimation {
                duration: animations.hover
                easing.type: animations.standardEasing
            }
        }
    }

    Popup {
        id: popup
        y: root.height + metrics.spacingTiny
        width: root.width
        modal: false
        focus: true
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside
        padding: metrics.spacingTiny

        enter: Transition {
            NumberAnimation {
                property: "opacity"
                from: 0
                to: 1
                duration: animations.popup
                easing.type: animations.standardEasing
            }
        }

        exit: Transition {
            NumberAnimation {
                property: "opacity"
                from: 1
                to: 0
                duration: animations.press
                easing.type: animations.standardEasing
            }
        }

        background: Rectangle {
            radius: metrics.radiusMedium
            color: theme.panelRaisedFill
            border.width: 1
            border.color: theme.separator
        }

        contentItem: Column {
            spacing: metrics.spacingTiny

            Repeater {
                model: root.options

                delegate: ItemDelegate {
                    id: optionDelegate

                    required property string modelData

                    width: popup.availableWidth
                    text: modelData
                    hoverEnabled: true
                    leftPadding: metrics.spacingSmall
                    rightPadding: metrics.spacingSmall

                    contentItem: Text {
                        text: optionDelegate.modelData
                        color: theme.black
                        font.family: typography.bodyFamily
                        font.pixelSize: typography.caption
                        elide: Text.ElideRight
                        verticalAlignment: Text.AlignVCenter
                    }

                    background: Rectangle {
                        radius: metrics.radiusSmall
                        color: optionDelegate.hovered ? theme.hoverFill : Qt.rgba(1, 1, 1, 0)
                    }

                    onClicked: {
                        root.optionSelected(optionDelegate.modelData)
                        popup.close()
                    }
                }
            }
        }
    }
}
