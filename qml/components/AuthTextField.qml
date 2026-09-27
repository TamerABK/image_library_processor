import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Control {
    id: root

    property alias text: input.text
    property string label: ""
    property string placeholderText: ""
    property bool passwordMode: false
    property string errorText: ""

    signal accepted(string text)

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }
    Room36Theme.Animations { id: animations }

    implicitWidth: metrics.authFieldWidth
    implicitHeight: metrics.authFieldHeight + (errorText.length > 0 ? errorLabel.implicitHeight + metrics.spacingTiny : 0)

    contentItem: ColumnLayout {
        spacing: metrics.spacingTiny

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: metrics.authFieldHeight
            radius: metrics.authFieldRadius
            color: theme.authBackground
            border.width: input.activeFocus ? 2 : 1
            border.color: input.activeFocus ? theme.focusRing : theme.mainBlue

            TextField {
                id: input
                anchors.fill: parent
                anchors.leftMargin: metrics.spacingMedium
                anchors.rightMargin: metrics.spacingMedium
                background: Item {}
                placeholderText: root.label.length > 0 ? root.label : root.placeholderText
                placeholderTextColor: theme.mainBlue
                color: theme.mainBlue
                font.family: typography.bodyFamily
                font.pixelSize: typography.caption
                selectByMouse: true
                verticalAlignment: TextInput.AlignVCenter
                echoMode: root.passwordMode ? TextInput.Password : TextInput.Normal
                leftPadding: 0
                rightPadding: 0
                topPadding: 0
                bottomPadding: 0

                onAccepted: root.accepted(text)
            }

            Behavior on border.color {
                ColorAnimation {
                    duration: animations.hover
                    easing.type: animations.standardEasing
                }
            }
        }

        Text {
            id: errorLabel
            Layout.fillWidth: true
            visible: root.errorText.length > 0
            text: root.errorText
            color: theme.midBlue
            font.family: typography.captionFamily
            font.pixelSize: typography.detail
            elide: Text.ElideRight
        }
    }
}
